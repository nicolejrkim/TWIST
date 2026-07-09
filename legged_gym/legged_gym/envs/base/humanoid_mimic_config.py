from .humanoid_char_config import HumanoidCharCfg
from .base_config import BaseConfig


class HumanoidMimicCfg(HumanoidCharCfg):
    class env(HumanoidCharCfg.env):
        enable_early_termination = True
        pose_termination = False
        pose_termination_dist = 1.0
        root_tracking_termination_dist = 0.7
        enable_tar_obs = False
        tar_obs_steps = [1]
        rand_reset = True
        ref_char_offset = 0.0
        global_obs = True
        track_root = True
        dof_err_w = None

        # reference root pose observation (extension): feed the policy the reference
        # root xy + yaw as an error relative to the robot's current root, expressed in
        # the robot's heading frame (3 dims per target obs step). The robot has no
        # observation of its own world xy/yaw, so an absolute reference would be
        # uninformative; the error form matches the deploy-side odometry signal.
        obs_ref_root_pose = False
        ref_root_pose_err_clip = 1.0  # [m] clip on the xy error obs, keeps values in-distribution
        # inject an initial robot-vs-reference pose error at reset so the policy sees
        # nonzero errors to close (otherwise error only accrues from drift and pushes)
        randomize_init_root_pose_err = False
        init_root_xy_err_range = 0.15  # [m], uniform +-
        init_root_yaw_err_range = 0.2  # [rad], uniform +-
        

class HumanoidMimicCfgPPO(BaseConfig):
    seed = 1
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        tanh_encoder_output = False
        fix_action_std = False
        obs_context_len = 0
        
    class algorithm:
        # training params
        grad_penalty_coef = 0.0
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4 #1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.008
        max_grad_norm = 1.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
        normalizer_update_iterations = 3000

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunner'
        num_steps_per_env = 24 # per iteration
        max_iterations = 20000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        