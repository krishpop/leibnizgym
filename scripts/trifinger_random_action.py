"""
@brief      Demo script for checking tri-finger environment.
"""

# leibnizgym
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
# python
from isaacgym import gymtorch
import torch


def compute_torque(env, Ji_t, ftip_force):
    env._gym.refresh_jacobian_tensors(env._sim)
    torque = 0
    for fid, frame_id in enumerate(env._fingertips_handles.values()):
        Ji = Ji_t[:, frame_id - 1, :3]
        Ji_T = Ji.transpose(1, 2)
        F = ftip_force[:, 3 * fid: 3 * fid + 3]
        torque += torch.matmul(Ji_T, F)
    return torque.squeeze()


def random_ftip_force(env):
    xy_ac = ((5 * torch.randn(env.get_action_shape(), dtype=torch.float, device=env.device))
             * torch.tensor([1, 1, 0] * 3, device=env.device, dtype=torch.float))
    z_ac = ((2 * torch.randn(env.get_action_shape(), dtype=torch.float, device=env.device) - 1)
            * torch.tensor([0, 0, 1] * 3, device=env.device, dtype=torch.float))
    ac = xy_ac + z_ac
    # ac = torch.ones(n_envs, dtype=torch.float, device=env.device)
    # ac += torch.tensor([0, 0, 1] * 3, dtype=torch.float, device=env.device).tile((env.get_action_shape()[0], 1))
    ac = ac.view(-1, 9, 1)
    return ac


def random_ftip_pos(env):
    xy_ac = ((torch.randn(env.get_action_shape(), dtype=torch.float, device=env.device)) * torch.tensor(
        [.1, .1, 0] * 3, device=env.device, dtype=torch.float)).reshape((-1, 9, 1))
    z_ac = ((torch.randn(env.get_action_shape(), dtype=torch.float, device=env.device)) * torch.tensor(
        [0, 0, .01] * 3, device=env.device, dtype=torch.float)).reshape((-1, 9, 1))
    ac = xy_ac + z_ac + env._fingertips_frames_state_history[0][:, :, :3].reshape((-1, 9, 1))
    # ac = torch.ones(n_envs, dtype=torch.float, device=env.device)
    # ac += torch.tensor([0, 0, 1] * 3, dtype=torch.float, device=env.device).tile((env.get_action_shape()[0], 1))
    ac = ac.view(-1, 9, 1)
    return ac


def main(use_ftip_f=False, use_ftip_pos=False, n=128, command_mode='torque', visualize=False):
    # configure the environment
    command_mode = 'torque' if use_ftip_f else command_mode
    env_config = {
        'num_instances': n,
        'aggregrate_mode': True,
        'control_decimation': 1,
        'command_mode': command_mode,
        'sim': {
            "use_gpu_pipeline": True,
            "physx": {
                "use_gpu": False
            }
        }
    }
    # create environment
    env = TrifingerEnv(config=env_config, device='cuda:0', verbose=True, visualize=visualize)
    # if use_ftip:
        # _jacobian = env._gym.acquire_jacobian_tensor(env._sim, "robot")
        # jacobian = gymtorch.wrap_tensor(_jacobian)
    _ = env.reset()
    print_info("Trifinger environment creation successful.")

    # sample run
    while True:
        if use_ftip_f:
            # action = compute_torque(env, jacobian, random_ftip_force(env))
            action = env._compute_fingertips_force_torque(random_ftip_force(env))
        elif use_ftip_pos:
            action = env._compute_fingertips_ik_action(random_ftip_pos(env))
        else:
            # zero action agent
            action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        # step through physics
        _, _, _, _ = env.step(action)
        # render environment
        if visualize:
            env.render()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ftip_f', action='store_true')
    parser.add_argument('--use_ftip_pos', action='store_true')
    parser.add_argument('--command_mode', choices=['position', 'torque'],
                        default='torque')
    parser.add_argument('--n', default=128, type=int)
    parser.add_argument('--v', action='store_true')
    args = parser.parse_args()
    main(args.use_ftip_f, args.use_ftip_pos, args.n, args.command_mode,
         args.v)


# EOF
