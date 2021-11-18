import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from diffcp import SolverError
import functools

from isaacgym.torch_utils import *

import torch
from cvxpylayers.torch import CvxpyLayer


class BatchForceOptProblem:
    def __init__(
        self,
        obj_mu=1.0,
        mass=0.016,
        gravity=-9.81,
        target_n=0.0,
        cone_approx=True,
        device="cuda:0",
    ):
        self.obj_mu = obj_mu
        self.mass = mass
        self.gravity = gravity
        self.target_n = target_n  # 1.0
        self.cone_approx = cone_approx
        self.initialized = False
        self.prob = None
        assert device in ["cuda:0", "cpu"], f'{device} not in ["cuda:0", "cpu"]'
        if device == "cuda:0" and not torch.cuda.is_available():
            print("switching device to CPU because cuda not available")
            device = "cpu"
        self.device = device

    def setup_cvxpy_layer(self):
        # Try solving optimization problem
        # contact force decision variable
        target_n = np.array([self.target_n, 0, 0] * 3).astype("float32", copy=False)
        self.target_n_t = torch.as_tensor(
            target_n, dtype=torch.float32, device=self.device
        )
        self.target_n_cp = cp.Parameter((9,), name="target_n", value=target_n)
        self.L = cp.Variable(9, name="l")
        self.W = cp.Parameter((6,), name="w_des")
        self.G = cp.Parameter((6, 9), name="grasp_m")
        cm = np.vstack((np.eye(3), np.zeros((3, 3)))) * self.mass

        inputs = [self.G, self.W, self.target_n_cp]
        outputs = [self.L]
        # self.Cm = cp.Parameter((6, 3), value=cm*self.mass, name='com')

        f_g = np.array([0, 0, self.gravity])
        w_ext = self.W + cm @ f_g

        f = self.G @ self.L - w_ext  # generated contact forces must balance wrench

        # Objective function - minimize force magnitudes
        contact_force = self.L - self.target_n_cp
        cost = cp.sum_squares(contact_force)

        # Friction cone constraints; >= 0
        self.constraints = []
        self.cone_constraints = []
        if self.cone_approx:
            self.cone_constraints += [cp.abs(self.L[1::3]) <= self.obj_mu * self.L[::3]]
            self.cone_constraints += [cp.abs(self.L[2::3]) <= self.obj_mu * self.L[::3]]
        else:
            self.cone_constraints.append(
                cp.SOC(self.obj_mu * self.L[::3], (self.L[2::3] + self.L[1::3])[None])
            )
        self.constraints.append(f == np.zeros(f.shape))

        self.prob = cp.Problem(
            cp.Minimize(cost), self.cone_constraints + self.constraints
        )
        self.policy = CvxpyLayer(self.prob, inputs, outputs)
        self.initialized = True

    def balance_force_test(self, des_wrench, balance_force, cp_list):
        weight = np.vstack([np.eye(3), np.zeros((3, 3))]) @ np.array(
            [0, 0, self.gravity * self.mass]
        )
        G = self.get_grasp_matrix(cp_list)
        w_ext = des_wrench + weight
        f = G @ balance_force - w_ext
        return f

    def __call__(self, des_wrench_t, cp_list):
        if not self.initialized:
            self.setup_cvxpy_layer()
        return self.run_fop(des_wrench_t, cp_list)

    def run_fop(self, des_wrench_t, cp_list):
        G_t = self.get_grasp_matrix(cp_list)
        n = len(cp_list)
        target_n_t = self.target_n_t.tile((n, 1))
        inputs = [G_t, des_wrench_t, target_n_t]
        try:
            (balance_force,) = self.policy(*inputs)
            return balance_force
        except SolverError:
            return torch.zeros((n, 9), device=cp_list.device)

    def get_grasp_matrix(self, cp_list):
        GT_list = []
        n = len(cp_list)
        H = _get_H_matrix().unsqueeze(0).tile((n, 1, 1))  # n x 9 x 18

        for i in range(3):
            cp_wf = cp_list[:, i]
            GT_i = self._get_grasp_matrix_single_cp(cp_wf)
            GT_list.append(GT_i)
        GT_full = torch.cat(GT_list, dim=1)  # n x 18 x 6
        GT = torch.bmm(H, GT_full)  # n x 9 x 6
        return GT.transpose(1, 2)  # n x 6 x 9

    def _get_grasp_matrix_single_cp(self, cp_wf):
        P = self._get_P_matrix(cp_wf[:, :3])  # n x 6 x 6
        quat_c_2_w = cp_wf[:, 3:]

        # Orientation of cp frame w.r.t. world frame
        # quat_c_2_w = quat_o_2_w * quat_c_2_o
        # R is rotation matrix from contact frame i to world frame
        R = euler_angles_to_matrix(torch.stack(get_euler_xyz(quat_c_2_w), dim=-1))
        _R0 = torch.zeros_like(R)  # n x 3 x 3
        R_bar = torch.cat((torch.cat([R, _R0], dim=-1),
                           torch.cat([_R0, R], dim=-1)), dim=-2)
        G = torch.bmm(P, R_bar)  # n x 6 x 6
        return G.transpose(1, 2)

    def _get_P_matrix(self, pos_wf):
        pos_wf = pos_wf.unsqueeze(1)  # shape: n x 1 x 3
        n = len(pos_wf)
        S = _skew_mat_t(pos_wf)  # shape: n x 3 x 3
        _I3 = torch.eye(3, dtype=pos_wf.dtype, device=pos_wf.device).tile((n, 1, 1))
        S = torch.cat((S, _I3), dim=2)
        _I6 = torch.eye(6, dtype=torch.float, device=pos_wf.device)[:3].tile(n, 1, 1)
        P = torch.cat((_I6, S), dim=1)  # shape: n x 6 x 6
        return P


def _skew_mat_t(x_vec):
    assert len(x_vec.shape) == 3
    n = len(x_vec)
    W_row0 = to_torch([[0, 0, 0, 0, 0, 1, 0, -1, 0]] * n, device=x_vec.device).view(n, 3, 3)
    W_row1 = to_torch([[0, 0, -1, 0, 0, 0, 1, 0, 0]] * n, device=x_vec.device).view(n, 3, 3)
    W_row2 = to_torch([[0, 1, 0, -1, 0, 0, 0, 0, 0]] * n, device=x_vec.device).view(n, 3, 3)
    x_skewmat = torch.cat(
        [
            torch.matmul(x_vec, W_row0.transpose(1, 2)),
            torch.matmul(x_vec, W_row1.transpose(1, 2)),
            torch.matmul(x_vec, W_row2.transpose(1, 2)),
        ],
        dim=1,
    )
    return x_skewmat


def _get_H_matrix():
    H_i = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    H = block_diag(H_i, H_i, H_i)
    return to_torch(H)


# from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L185


def euler_angles_to_matrix(euler_angles):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    convention = "XYZ"
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
