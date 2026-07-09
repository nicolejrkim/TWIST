from isaacgym.torch_utils import *

import torch

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .g1_mimic_distill_config import G1MimicPrivCfg, G1MimicStuCfg
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos

def g1_body_from_38_to_52(body_pos_38: torch.Tensor) -> torch.Tensor:
    """
    将形状 (N, 38, 3) 的关节坐标转换为形状 (N, 52, 3)。
    多出来的手指等关节在输出中将填充为 (0, 0, 0)。

    参数:
    -------
        body_pos_38 : torch.Tensor
            大小为 (N, 38, 3) 的关节坐标，N 为批量大小

    返回:
    -------
        body_pos_52 : torch.Tensor
            大小为 (N, 52, 3) 的关节坐标
    """

    # 构建一个大小为 52 的整型索引张量，用于说明：
    # “52-link 的每个关节，对应到 38-link 中的哪一个下标？”
    # 若对应不到（例如手指关节），则为 -1。
    idx_map_52_list = [-1] * 52
    # 直接在列表中指定 38->52 的映射：
    # 0~29 不变, 30->37, 31->38, 32->39, 33->40, 34->41, 35->42, 36->43, 37->44
    # ---------------------------------------------------------------------
    idx_map_52_list[0]  = 0   # pelvis
    idx_map_52_list[1]  = 1
    idx_map_52_list[2]  = 2
    idx_map_52_list[3]  = 3
    idx_map_52_list[4]  = 4
    idx_map_52_list[5]  = 5
    idx_map_52_list[6]  = 6
    idx_map_52_list[7]  = 7
    idx_map_52_list[8]  = 8
    idx_map_52_list[9]  = 9
    idx_map_52_list[10] = 10
    idx_map_52_list[11] = 11
    idx_map_52_list[12] = 12
    idx_map_52_list[13] = 13
    idx_map_52_list[14] = 14
    idx_map_52_list[15] = 15
    idx_map_52_list[16] = 16
    idx_map_52_list[17] = 17
    idx_map_52_list[18] = 18
    idx_map_52_list[19] = 19
    idx_map_52_list[20] = 20
    idx_map_52_list[21] = 21
    idx_map_52_list[22] = 22
    idx_map_52_list[23] = 23
    idx_map_52_list[24] = 24
    idx_map_52_list[25] = 25
    idx_map_52_list[26] = 26
    idx_map_52_list[27] = 27
    idx_map_52_list[28] = 28
    idx_map_52_list[29] = 29
    idx_map_52_list[37] = 30
    idx_map_52_list[38] = 31
    idx_map_52_list[39] = 32
    idx_map_52_list[40] = 33
    idx_map_52_list[41] = 34
    idx_map_52_list[42] = 35
    idx_map_52_list[43] = 36
    idx_map_52_list[44] = 37
    # 其余下标(手指相关)依旧保持 -1

    # 转换成 PyTorch 张量，放到和输入相同的 device 上
    idx_map_52 = torch.tensor(idx_map_52_list, 
                              dtype=torch.long, 
                              device=body_pos_38.device)

    # 创建输出张量，大小 (N, 52, 3)，默认填零
    N = body_pos_38.shape[0]
    body_pos_52 = torch.zeros((N, 52, 3), 
                              dtype=body_pos_38.dtype, 
                              device=body_pos_38.device)

    # 构建一个布尔掩码，筛选出 idx_map_52 >= 0 的关节
    valid_mask = (idx_map_52 >= 0)

    # 对有效关节通过高级索引直接复制，无需对 N 进行循环
    body_pos_52[:, valid_mask, :] = body_pos_38[:, idx_map_52[valid_mask], :]

    return body_pos_52



class G1MimicDistill(HumanoidMimic):
    def __init__(self, cfg: G1MimicPrivCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == 'student':
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000
            # self.motion_difficulty = torch.ones_like(self.motion_difficulty)

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
        
        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos[env_ids].shape[1]:
            body_pos = g1_body_from_38_to_52(body_pos)
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            body_pos = g1_body_from_38_to_52(body_pos)
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
    def _update_motion_difficulty(self, env_ids):
        if self.obs_type == 'priv':
            super()._update_motion_difficulty(env_ids)
        elif self.obs_type == 'student':
            super()._update_motion_difficulty(env_ids) # currently we use the same strategy for student
        else:
            return

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
    
    def _init_buffers(self):
        super()._init_buffers()
        self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_obs_single), device=self.device)
        self.privileged_obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single), device=self.device)
    
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        
        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim:ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, ang_vel_dim+imu_dim:ang_vel_dim+imu_dim+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, ang_vel_dim+imu_dim+self.num_dof:ang_vel_dim+imu_dim+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        
        return noise_scale_vec
            
    def _get_mimic_obs(self):
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_obs_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])

        # teacher v0
        priv_mimic_obs_terms = [
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
            whole_key_body_pos, # num_bodies * 3 dims
        ] # shape: (num_envs, num_steps, 8 + num_dof + num_key_bodies * 3)


        # v6, align mocap
        mimic_obs_terms = [
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
        ] # shape: (num_envs, num_steps, 8 + num_dof)

        if self._obs_ref_root_pose:
            # reference root pose relative to the robot's current root, in the robot's
            # heading frame. Motion frames live in the motion-file frame; adding
            # episode_init_origin maps them to the sim world (same convention as
            # _update_ref_motion). Appended last so all legacy obs indices are unchanged.
            ref_xy_world = root_pos[..., 0:2] + self.episode_init_origin[:, None, 0:2]
            err_xy_world = ref_xy_world - self.root_states[:, None, 0:2]
            # self.yaw is stale for envs reset this step, so recompute from root_states
            _, _, robot_yaw = euler_from_quaternion(self.root_states[:, 3:7])
            cos_yaw = torch.cos(robot_yaw)[:, None]
            sin_yaw = torch.sin(robot_yaw)[:, None]
            err_xy_local = torch.stack((
                cos_yaw * err_xy_world[..., 0] + sin_yaw * err_xy_world[..., 1],
                -sin_yaw * err_xy_world[..., 0] + cos_yaw * err_xy_world[..., 1],
            ), dim=-1)
            err_xy_local = torch.clamp(err_xy_local, -self._ref_root_pose_err_clip, self._ref_root_pose_err_clip)
            yaw_diff = yaw - robot_yaw[:, None, None]
            err_yaw = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            ref_root_pose_err = torch.cat((err_xy_local, err_yaw), dim=-1) # (num_envs, num_steps, 3)
            priv_mimic_obs_terms.append(ref_root_pose_err) # 3 dims
            mimic_obs_terms.append(ref_root_pose_err) # 3 dims

        priv_mimic_obs_buf = torch.cat(priv_mimic_obs_terms, dim=-1)
        mimic_obs_buf = torch.cat(mimic_obs_terms, dim=-1)[:, 0:1] # first target step only

        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()
        
        proprio_obs_buf = torch.cat((
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        
        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        else:
            proprio_obs_buf += 0.
        dof_vel_start_dim = 5 + self.dof_pos.shape[1]

        # disable ankle dof
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1) # shape: (num_envs, num_key_bodies * 3)
        
        if self.cfg.domain_rand.domain_rand_general:
            priv_info = torch.cat((
                self.base_lin_vel, # 3 dims
                self.root_states[:, 2:3], # 1 dim
                key_body_pos, # num_bodies * 3 dims
                self.contact_forces[:, self.feet_indices, 2] > 5., # 2 dims, foot contact
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
            ), dim=-1)
        else:
            priv_info = torch.zeros((self.num_envs, self.cfg.env.n_priv_info), device=self.device)
        
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        self.privileged_obs_buf = priv_obs_buf
        
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student':
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        if self.cfg.env.history_len > 0:
            self.privileged_obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.privileged_obs_history_buf[:, 1:],
                    priv_obs_buf.unsqueeze(1)
                ], dim=1)
            )
            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == 'student':
                self.obs_history_buf = torch.where(
                    (self.episode_length_buf <= 1)[:, None, None], 
                    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                    torch.cat([
                        self.obs_history_buf[:, 1:],
                        obs_buf.unsqueeze(1)
                    ], dim=1)
                )


############################################################################################################
##################################### Extra Reward Functions################################################
############################################################################################################

    def _reward_waist_dof_acc(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, waist_dof_idx], dim=1)
    
    def _reward_waist_dof_vel(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square(self.dof_vel[:, waist_dof_idx]), dim=1)
    
    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
    
    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)
    
    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1)
