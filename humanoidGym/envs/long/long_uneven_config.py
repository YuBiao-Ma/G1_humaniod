from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from humanoidGym import GYM_ROOT_DIR

class LongUnevenRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.122] # x,y,z [m]
        
        default_joint_angles = {
            'J_hip_r_roll': 0.0,
            'J_hip_r_yaw': 0.,
            'J_hip_r_pitch': 0.305913,
            'J_knee_r_pitch': -0.670418,
            'J_ankle_r_pitch': 0.371265,
            'J_ankle_r_roll': 0.0,

            'J_hip_l_roll': 0.0,
            'J_hip_l_yaw': 0.,
            'J_hip_l_pitch': 0.305913,
            'J_knee_l_pitch': -0.670418,
            'J_ankle_l_pitch': 0.371265,
            'J_ankle_l_roll': 0.0,
        }
        
        target_joint_angles = {
            'J_hip_r_roll': 0.,
            'J_hip_r_yaw': 0.,
            'J_hip_r_pitch': 0.305913,
            'J_knee_r_pitch': -0.670418,
            'J_ankle_r_pitch': 0.371265,
            'J_ankle_r_roll': 0.,

            'J_hip_l_roll': 0.,
            'J_hip_l_yaw': 0.,
            'J_hip_l_pitch': 0.305913,
            'J_knee_l_pitch': -0.670418,
            'J_ankle_l_pitch': 0.371265,
            'J_ankle_l_roll': 0.,
        }
        
    class env(LeggedRobotCfg.env):
        num_single_observations = 45 + 3 + 2 + 1#+ 187# add 3 for base linvel to provide correct regression target, sin cos
        num_critic_single_observations = 50 + 19 + 6 + 1
       
        num_actions = 12
        num_obs_lens = 51#51#21#21#66#10#20
        critic_num_obs_lens = 5
        num_observations = num_obs_lens * num_single_observations
        num_privileged_obs = num_critic_single_observations*critic_num_obs_lens + 187 #98 + 187 + 2
        
        num_envs = 4096

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        
        randomize_base_mass = True
        added_mass_range = [-2., 5.]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        
        randomize_com = True
        com_x = [-0.1,0.1]
        com_y = [-0.1,0.1]
        com_z = [-0.1,0.1]
        
        randomize_motor_strength = True
        motor_strength = [0.8,1.2]
        
        randomize_gains = True
        kp_range = [0.8,1.2]
        kd_range = [0.8,1.2]
        
        add_action_lag = True
        action_lag_timesteps_range = [0,30]
        
        randomize_restitution = False
        restitution_range = [0.0,1.0]
        
        randomize_inertia = True
        randomize_inertia_range = [0.7, 1.5]
        # randomize_inertia_range = [0.9, 1.1]
        
        randomize_init_joint_scale = True
        init_joint_scale = [0.5,1.5]
        
        randomize_init_joint_offset = True
        init_joint_offset = [-0.1,0.1]
        
        randomize_rfi = True
        rfi_ep = [-0.1,0.1]
        rfi_st = [-0.1,0.1]
         
        randomize_motor_zero_offset = False
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_joint_friction = False
        joint_friction_range = [0.8, 1.2]

        randomize_joint_damping = False
        joint_damping_range = [0.5, 2.5]

        randomize_joint_armature = False
        joint_armature_range = [0.8, 1.2]    
        
        
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = True
        # rough terrain only:
        measure_heights = True
        # static_friction = 0.6
        # dynamic_friction = 0.6
        
        static_friction = 0.1
        dynamic_friction = 0.1
        
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        
        terrain_dict = {"flat": 0.05, 
                        "rough flat": 0.05,
                        "slope up": 0.075,
                        "slope down": 0.075, 
                        "rough slope up": 0.075,
                        "rough slope down": 0.075, 
                        "stairs up": 0.20, 
                        "stairs down": 0.10,
                        "discrete": 0.2}
        
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.4]   # rad
        rough_slope_range = [0.0, 0.05]
        # stair_width_range = [0.50, 0.40]
        stair_width_range = [0.35, 0.30]
        stair_height_range = [0.10, 0.20]
        # stair_height_range = [0.15, 0.20]
        discrete_height_range = [0.05, 0.10]
        restitution = 0.
        fractal_noise_strength = 0.05
        
        # measured_points_x = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        
        stiffness = {
            'J_hip_r_roll': 400.,
            'J_hip_r_yaw': 200.,
            'J_hip_r_pitch': 400.,
            'J_knee_r_pitch': 400.,
            'J_ankle_r_pitch': 120.,
            'J_ankle_r_roll': 120.,

            'J_hip_l_roll': 400.,
            'J_hip_l_yaw': 200.,
            'J_hip_l_pitch': 400.,
            'J_knee_l_pitch': 400.,
            'J_ankle_l_pitch': 120.,
            'J_ankle_l_roll': 120.,
        }
        damping = {
            'J_hip_r_roll': 2.,
            'J_hip_r_yaw': 2.,
            'J_hip_r_pitch': 2.,
            'J_knee_r_pitch': 4.,
            'J_ankle_r_pitch': 0.5,
            'J_ankle_r_roll': 0.5,

            'J_hip_l_roll': 2.,
            'J_hip_l_yaw': 2.,
            'J_hip_l_pitch': 2.,
            'J_knee_l_pitch': 4.,
            'J_ankle_l_pitch': 0.5,
            'J_ankle_l_roll': 0.5,
        }
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5#1#0.5#1#0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20
        use_filter = True
        exp_avg_decay = 0.05
        

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/humanoidGym/resources/robots/mujoco_long/loong_knee_collision.urdf'
        #loong_four_ball_collision
        # file = '{LEGGED_GYM_ROOT_DIR}/humanoidGym/resources/robots/mujoco_long/loong_four_ball_collision.urdf'
        name = "long"
        foot_name = ['Link_ankle_l_roll', 'Link_ankle_r_roll']
        knee_name = ['Link_knee_l_pitch', 'Link_knee_r_pitch']
        penalize_contacts_on = ["Link_knee_l_pitch", "Link_knee_r_pitch"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = False
        obs_mirror = ["base_lin_vel","commands","phase","stand_cmd","base_ang_vel","projected_gravity","dofs","dofs","dofs"]
        # obs_mirror = ["base_lin_vel","commands","base_ang_vel","projected_gravity","dofs","dofs","dofs"]
    
    class commands:
        curriculum = False
        max_curriculum = 1
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True# if true: compute ang vel command from heading error
        
        stand_com_threshold = 0.1#0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True# use stand_com_threshold or not
        
        class ranges:
            lin_vel_x = [-0.8, 1.0] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        
        base_height_target = 1.05
        only_positive_rewards = True
        
        target_joint_pos_scale = 0.30#0.26#0.165#0.26#0.165
        # local frame   
        target_feet_height = -0.98#-1.0    
        cycle_time = 0.8
        max_contact_force = 1500
        tracking_sigma = 5
        
        min_dist = 0.20
        max_dist = 0.8
        #max_dist = 0.40                
        
        class scales( LeggedRobotCfg.rewards.scales ):
            
            joint_pos = 2.0#3.0
            feet_clearance = 1.0#0.5
            feet_contact_number = 1.0
            
            #no_fly = 0.3
            no_fly = 0.5
            # joint_pos = 4.0
            # feet_clearance = 0.8
            # feet_contact_number = 3.0
            # joint_pos = 1.5
            # feet_clearance = 1.0#0.5
            # feet_contact_number = 0.5
            # gait
            feet_air_time = 1.5#2.0#1.5 # new default angle
            # feet_air_time = 2.5 # origin default
            
            foot_slip = -0.5#-0.4#-0.2#-0.1
            feet_distance = 0.2
            knee_distance = 0.2
            simple_feet_min_distance = 10
            simple_knee_min_distance = 10
            feet_rotation = 0.4
            # contact 
            #feet_contact_forces = -0.02#-0.01
            # vel tracking
            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 1.0#0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            
            # stand_still_pen = -10
            # stand_still = 2#2
            # idol_root_vel = -10
            # idol_action_smoothness = -0.05#0.05
            # stand_still_power = -0.1 #-1e-2
            # stand_still_power = -1e-3
            # idol_feet_contact = 10
            # idol_base_height = 1
            
            # base pos
            default_joint_pos = 2#2#2#1#0.8
            orientation = 1.
            #base_height = 0.2
            base_height = 0.05#0.1#0.2
            #base_height = 0.05
            # base_height_uneven = -0.1
            base_acc = 0.4#0.2
            # energy
            #action_smoothness = -0.005
            action_smoothness = -0.02
            # hip_yaw_roll_action_smoothness = -0.02
            hip_yaw_action_smoothness = -0.02
            hip_roll_action_smoothness = -0.02
            ankle_pitch_action_smoothness = -0.02
            ankle_roll_action_smoothness = -0.02
            # exp_energy = 0.05
          
            torques = -0.00001
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            
            stumble = -3.0
            # power_dist = -1e-7
            
            #exp_energy = 0.02
            
            # foot_normal_reward = 0.2#0.05
            # feet_height_var = 0.2
            foot_normal_reward = 0.05#0.1#0.05
            feet_height_var = 0.5
            
            # hip_roll_default = 0.2
            # hip_roll_energy = 0.2
            ankle_energy = 0.2
            hip_roll_yaw_energy = 0.4
            
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation_pen = -1.0
            contact_momentum = -1e-4
            foot_landing_vel = -0.1
            
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1
            # ankle_torque_limits = -0.1
            # hip_roll_yaw_torque_limits = -0.1

            power = -1e-5
            
            collision = -1
            
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.02#0.01
            dof_vel = 1.5#5.0#1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class sim:
        dt =  0.001
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class LongUnevenRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-3
        max_grad_norm = 1
        
        #    #Random Network Distillation
        # class rnd_cfg:
        #     weight= 1 # initial weight of the RND reward

        #     # note: This is a dictionary with a required key called "mode" which can be one of "constant" or "step".
        #     #   - If "constant", then the weight is constant.
        #     #   - If "step", then the weight is updated using the step scheduler. It takes additional parameters:
        #     #     - max_num_steps: maximum number of steps to update the weight
        #     #     - final_value: final value of the weight
        #     # If None, then no scheduler is used.
        #     weight_schedule=None

        #     reward_normalization=False  # whether to normalize RND reward
        #     gate_normalization=True  # whether to normalize RND gate observations

        #     # -- Learning parameters
        #     learning_rate=0.001  # learning rate for RND

        #     # -- Network parameters
        #     # note: if -1, then the network will use dimensions of the observation
        #     num_outputs=1  # number of outputs of RND network
        #     predictor_hidden_dims = [256,128] # hidden dimensions of predictor network
        #     target_hidden_dims = [256,128]  # hidden dimensions of target network
        
        #    # -- Symmetry Augmentation
        class symmetry_cfg:
            use_data_augmentation=False  # this adds symmetric trajectories to the batch
            use_mirror_loss=True  # this adds symmetry loss term to the loss function
            # coefficient for symmetry loss term
            # if 0, then no symmetry loss is used
            mirror_loss_coeff=1.0
            
            
    class runner( LeggedRobotCfgPPO.runner ):
        empirical_normalization = True
        policy_class_name = "ActorCritic"
        max_iterations = 30000
        run_name = 'long_reward_barlowtwin_baseline'
        experiment_name = 'g1'
        resume = False
        
        

  
