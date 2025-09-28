import sys
from humanoidGym import GYM_ROOT_DIR
import os
import cv2
from isaacgym import gymapi
import isaacgym
from humanoidGym.envs import *
from humanoidGym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 3)
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 1
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.add_action_lag = False
    env_cfg.domain_rand.randomize_rfi = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_init_joint_offset = False
    env_cfg.domain_rand.randomize_init_joint_scale = False
    env_cfg.domain_rand.randomize_inertia = False

    env_cfg.env.episode_length_s = 20

    env_cfg.env.test = True
    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.heading = [0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, infos = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device="cpu")
    if GET_LATENT:
        get_latent = ppo_runner.get_latents(device="cpu")

    # export policy as a jit module
    if EXPORT_POLICY:
        path = os.path.join(GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, ppo_runner.obs_normalizer)
        print('Exported policy as jit script to: ', path)

    # ====== rollout 配置 ======
    STEPS = 300
    REC_ENV_ID = 0  # 记录第0个环境
    SAVE_PREFIX = "pyramid_sloped_terrain"
    SAVE_CSV = True



    joint_order = list(env_cfg.init_state.default_joint_angles.keys())
    left_joints = [
        "joint_left_hip_pitch",
        "joint_left_hip_roll",
        "joint_left_hip_yaw",
        "joint_left_knee",
        "joint_left_ankle_pitch",
        "joint_left_ankle_roll",
    ]
    right_joints = [
        "joint_right_hip_pitch",
        "joint_right_hip_roll",
        "joint_right_hip_yaw",
        "joint_right_knee",
        "joint_right_ankle_pitch",
        "joint_right_ankle_roll",
    ]
    name_to_idx = {name: i for i, name in enumerate(joint_order)}

    buf = []
    total_steps = 10* int(env.max_episode_length)

    # set rgba camera sensor for debug and doudle check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.3, -3, 0.5)
    # camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_local_transform.r = gymapi.Quat.from_euler_zyx(
        0.0,   # yaw  (绕 z)
        np.deg2rad(-20),   # pitch(绕 y，负值=向下俯)
        np.deg2rad(90),              # roll (绕 x)
    ) 
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1980
    camera_props.height = 1980

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    video_duration = 60
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    # ====== 用于收集 latent ======
    latents_list = []
    pred_list = []

    for i in range(total_steps):
        actions = policy(obs.detach().to("cpu"))
     

        if GET_LATENT:
            latents, pred_class = get_latent(obs.detach().to("cpu"))
            latents_np = latents.detach().cpu().numpy()
            pred_np = pred_class.detach().cpu().numpy()
            latents_list.append(latents_np)
            pred_list.append(pred_np)

        obs, rews, dones, infos = env.step(actions.detach())
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)

        # 读取期望关节位置
        tdp = env.target_dof_pos
        if isinstance(tdp, torch.Tensor):
            tdp = tdp.detach().cpu().numpy()
        buf.append(tdp[REC_ENV_ID].copy())

        if PLOT and i == STEPS - 1:
            data = np.asarray(buf)  # (steps, dof)
            steps, dof = data.shape
            fig, axes = plt.subplots(6, 2, figsize=(10, 12))
            x = np.arange(steps)
            for row, name in enumerate(left_joints):
                j = name_to_idx[name]
                ax = axes[row, 0]
                ax.plot(x, data[:, j])
                ax.set_title(name)
                ax.set_xlabel("step")
                ax.set_ylabel("target_dof_pos [rad]")
            for row, name in enumerate(right_joints):
                j = name_to_idx[name]
                ax = axes[row, 1]
                ax.plot(x, data[:, j])
                ax.set_title(name)
                ax.set_xlabel("step")
                ax.set_ylabel("target_dof_pos [rad]")
            fig.suptitle(f"Env {REC_ENV_ID} - target_dof_pos (first {steps} steps)")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

        if RECORD_FRAMES:
            img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((1980,1980,4))[:,:,:3]
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 
        # ====== 保存 CSV ======
        if GET_LATENT and SAVE_CSV and i == STEPS -1 > 0:
            latents_all = np.concatenate(latents_list, axis=0)  # [steps*num_envs, D]
            pred_all = np.concatenate(pred_list, axis=0)        # [steps*num_envs]
            df = pd.DataFrame(latents_all)
            df["pred_class"] = pred_all
            csv_path = f"{SAVE_PREFIX}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved latents & pred_class to {csv_path}, shape={latents_all.shape}")


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    PLOT = False
    GET_LATENT = False
    args = get_args()
    args.rl_device = 'cpu'
    play(args)
