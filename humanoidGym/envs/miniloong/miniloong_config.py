from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class MiniloongCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.75] # x,y,z [m]
        default_joint_angles = {
            'joint_left_hip_pitch': 0.39,   # [rad]
            'joint_left_hip_roll':  -0.0,   # [rad]
            'joint_left_hip_yaw':   -0.12,   # [rad]
            'joint_left_knee':  0.74,   # [rad]
            'joint_left_ankle_pitch': 0.36,   # [rad] 
            'joint_left_ankle_roll': 0,   # [rad]

            'joint_right_hip_pitch': 0.39,   # [rad]
            'joint_right_hip_roll':  -0.0,   # [rad]
            'joint_right_hip_yaw':   -0.12,   # [rad]
            'joint_right_knee':  0.74,   # [rad]
            'joint_right_ankle_pitch': 0.36,   # [rad]
            'joint_right_ankle_roll': 0,   # [rad]
        }
        target_joint_angles = {
            'joint_left_hip_pitch': 0.39,   # [rad]
            'joint_left_hip_roll':  -0.0,   # [rad]
            'joint_left_hip_yaw':   -0.12,   # [rad]
            'joint_left_knee':  0.74,   # [rad]
            'joint_left_ankle_pitch': 0.36,   # [rad] 
            'joint_left_ankle_roll': 0,   # [rad]

            'joint_right_hip_pitch': 0.39,   # [rad]
            'joint_right_hip_roll':  -0.0,   # [rad]
            'joint_right_hip_yaw':   -0.12,   # [rad]
            'joint_right_knee':  0.74,   # [rad]
            'joint_right_ankle_pitch': 0.36,   # [rad]
            'joint_right_ankle_roll': 0,   # [rad]
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
        friction_range = [0.4, 1.3]
        
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
        randomize_inertia_range = [0.9, 1.1]
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
        
        static_friction = 0.6
        dynamic_friction = 0.6
        
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
          # PD Drive parameters:
        stiffness = {
          
            'hip_pitch': 150, #100.,
            'hip_roll': 150, #100.,
            'hip_yaw': 150, #100.,            
            'knee': 150, #150.,
            'ankle_pitch': 40, #20.,
            'ankle_roll': 40, #20.,
        }  # [N*m/rad]
        damping = {
                 
            'hip_pitch': 5,# 1.2,
            'hip_roll': 5, # 1.2,
            'hip_yaw': 5, # 1.2,
            'knee': 5, # 2,
            'ankle_pitch': 2.5, #1,
            'ankle_roll': 2.5, #1,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20
        use_filter = True
        exp_avg_decay = 0.05

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/urdf/miniloong_v4_12dof_c_new_V41_1.urdf'
        name = "miniloong"
        foot_name = ["link_left_ankle_roll","link_right_ankle_roll"]
        knee_name = ["link_left_knee","link_right_knee"]
        # penalize_contacts_on = ["hip_yaw", "knee"]
        penalize_contacts_on = ["knee","hip"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        obs_mirror = ["base_lin_vel","commands","phase","stand_cmd","base_ang_vel","projected_gravity","dofs","dofs","dofs"]
    
    class commands:
        curriculum = True
        max_curriculum = 2
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.8, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
  
    class commands:
        curriculum = False
        max_curriculum = 1
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True# if true: compute ang vel command from heading error
        
        stand_com_threshold = 0.1#0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = False# use stand_com_threshold or not
        
        class ranges:
            lin_vel_x = [-0.8, 1.0] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        
        base_height_target = 0.75
        only_positive_rewards = True
        
        target_joint_pos_scale = 0.26#0.26#0.165#0.26#0.165
        # local frame   
        target_feet_height = -0.68#-1.0    
        cycle_time = 0.6
        max_contact_force = 500
        tracking_sigma = 5
        
        min_dist = 0.1
        # max_dist = 0.8
        max_dist = 0.3                
        
        class scales( LeggedRobotCfg.rewards.scales ):
           
            joint_pos = 0.5#1.0#2.0#3.0
            feet_clearance = 0.5#1.0
            feet_contact_number = 2.0#1.0#1.0
            no_fly = 0.5
            feet_air_time = 1.5#2.0
            
            foot_slip = -0.4#-0.1
            feet_distance = 0.2
            knee_distance = 0.2
            feet_rotation = 0.2

            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5
            low_speed = 0.2
            track_vel_hard = 0.5
            
            # base pos
            default_joint_pos = 1
            orientation = 1.0#1.0
            base_height = 0.2#0.05
            base_acc = 0.2
            # energy
            action_smoothness = -0.02
            hip_yaw_action_smoothness = -0.005
            hip_roll_action_smoothness = -0.005
            ankle_pitch_action_smoothness = -0.005
            ankle_roll_action_smoothness = -0.005
          
            torques = -0.00001
            power = -1e-5
            dof_vel = -1e-5
            dof_acc = -2.5e-7
            
            stumble = -5.0
            foot_normal_reward = 0.05
            feet_height_var = 0.5
            
            ankle_energy = 0.2
            hip_yaw_energy = 0.2
            knee_energy = 0.1
            hip_pitch_energy = 0.1
            
            default_joint_yaw = 0.5
            default_joint_ankle_roll = 0.5
            
            contact_momentum = -1e-4
            foot_landing_vel = -0.1
            
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1
            # ankle_pitch_limit = 10
            
            termination = -1#-10
            collision = -10
            
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


class MiniloongCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        class_name = 'TeacherPPO'
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
        policy_class_name = "ActorCritic_Teacher"
        algorithm_class_name = 'TeacherPPO'
        max_iterations = 30000
        run_name = 'long_reward_barlowtwin_baseline'
        experiment_name = 'miniloong_teacher_vq'
        resume = False