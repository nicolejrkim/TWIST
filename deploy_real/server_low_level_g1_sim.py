import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
from data_utils.params import DEFAULT_MIMIC_OBS
import os
from data_utils.rot_utils import quatToEuler

def draw_root_velocity(mujoco_model, mujoco_data, mujoco_viewer, tgt_root_vel, init_geom_id, root_name, rgba_velocity=[1, 1, 0, 1]):
    """
    Draws an arrow representing velocity, for debug/visualization.
    """
    mujoco_viewer.user_scn.ngeom = init_geom_id
    root_body_id = mujoco_model.body(root_name).id
    root_pos = mujoco_data.xpos[root_body_id]
    root_vel = tgt_root_vel
    vel_scale = 1.0

    mujoco.mjv_initGeom(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=rgba_velocity,
    )
    mujoco.mjv_connector(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=0.01,
        from_=root_pos,
        to=root_pos + vel_scale * np.array(root_vel),
    )
    mujoco_viewer.user_scn.ngeom += 1
    return mujoco_viewer.user_scn.ngeom


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------
def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    total_degrees = 33
    wrist_ids = [27, 32]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]
    wrist_dof_pos = mimic_obs[wrist_ids]

    return policy_target, wrist_dof_pos

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def aggregate_wrist_dof_pos(body_dof_pos, wrist_dof_pos):
    total_degrees = 25
    wrist_ids = [19, 24]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    whole_body_pd_target = np.zeros(total_degrees)
    whole_body_pd_target[other_ids] = body_dof_pos
    whole_body_pd_target[wrist_ids] = wrist_dof_pos
    
    return whole_body_pd_target
    
class RealTimePolicyController:
    def __init__(self,
                 xml_file,
                 policy_path,
                 device='cuda',
                 record_video=False,
                 video_path="debug_sim.mp4",
                 headless=False,
                 sim_duration=100000.0,
                 real_time_factor=1.0,
                 kp_recovery=0.0,
                 kp_yaw_recovery=0.0,
                 kd_recovery=0.0,
                 kd_yaw_recovery=0.0,
                 max_recovery_vel=0.5,
                 max_recovery_yaw_rate=1.0,
                 log_recovery=None,
                 push_force=0.0,
                 push_time=2.0,
                 push_duration=0.2,
                 push_angle_deg=90.0):

        # extension: outer-loop global-trajectory recovery (Kp on world pose error,
        # injected through the mimic obs velocity / yaw-rate channels)
        self.kp_recovery = kp_recovery
        self.kp_yaw_recovery = kp_yaw_recovery
        self.kd_recovery = kd_recovery
        self.kd_yaw_recovery = kd_yaw_recovery
        self.max_recovery_vel = max_recovery_vel
        self.max_recovery_yaw_rate = max_recovery_yaw_rate
        self.recovery_active = (kp_recovery > 0 or kp_yaw_recovery > 0
                                or kd_recovery > 0 or kd_yaw_recovery > 0)
        self.recovery_anchor = None  # SE(2) transform: motion-file frame -> sim world
        self.recovery_prev_err = None  # (t, err_xy, yaw_err) for the D term
        self.recovery_err_dot = np.zeros(3)  # EMA-filtered [ex_dot, ey_dot, eyaw_dot]
        if self.recovery_active:
            print(f"[Recovery] global recovery ON: kp_pos={kp_recovery}, kp_yaw={kp_yaw_recovery}, "
                  f"kd_pos={kd_recovery}, kd_yaw={kd_yaw_recovery}, "
                  f"max_vel={max_recovery_vel}, max_yaw_rate={max_recovery_yaw_rate}")

        # extension: per-step global-tracking log (for data_utils/compare_recovery.py)
        # and a scripted push, both timed relative to the first reference pose so
        # baseline and recovery runs are comparable
        self.log_recovery = log_recovery
        self.recovery_log = [] if log_recovery is not None else None
        self.push_force = push_force
        self.push_time = push_time
        self.push_duration = push_duration
        self.push_angle = np.deg2rad(push_angle_deg)
        self.motion_start_t = None  # sim time when the reference pose first appeared
        if self.push_force != 0.0:
            print(f"[Recovery] scripted push: {push_force} N for {push_duration}s "
                  f"at t={push_time}s after motion start, direction {push_angle_deg} deg")

        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device

        # Load policy
        self.policy = torch.jit.load(policy_path, map_location=device)
        print(f"Policy loaded from {policy_path}")

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.pelvis_body_id = self.model.body("pelvis").id
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            

        if headless:
            self.viewer = None
        else:
            self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            self.viewer.cam.distance = 2.0

        # Example defaults & placeholders
        self.num_actions = 23
        self.sim_duration = sim_duration
        self.sim_dt = 0.001
        # wall-clock budget per sim step; < 1.0 slows the sim so a motion server
        # paced with the same factor stays in sync (sim2sim clock alignment)
        self.wall_dt = self.sim_dt / real_time_factor
        self.sim_decimation = 20

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
        """
        mimic_obs = np.concatenate([
        root_pos[2:3],      # just the z for height
        rpy,                # roll, pitch, yaw
        root_vel_relative,  # local root vel
        dof_pos])
        """
        self.default_mimic_obs = DEFAULT_MIMIC_OBS["g1"]
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([0, 0, 0, 1]),
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.2, 0.0, 1.2, 0.0, # left arm (4)
                0.0, -0.2, 0.0, 1.2, 0.0, # right arm (4)
                ])
        ])
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 20,
                40, 40, 40, 40, 20,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 1,
                5, 5, 5, 5, 1,
            ])
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25, 25,
                25, 25, 25, 25, 25,
            ])
        
        self.action_scale = 0.5

        
        self.ankle_idx = [4, 5, 10, 11]
        
        # For multi-step history
        self.n_mimic_obs = 31
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3*self.num_actions
        self.proprio_history_buf = deque(maxlen=10)
        for _ in range(10):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))

        self.record_video = record_video
        self.video_path = video_path

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)
        
        body_ids = [0,1,2,3,4,5,
                    6,7,8,9,10,11,
                    12,13,14,
                    15,16,17,18,# 19
                    20,21,22,23, # 24
                    ]
        wrist_ids = [19, 24]
        
        whole_body_dof = qpos[7:]
        whole_body_dof_vel = qvel[6:]
        body_dof_pos = qpos[[f+7 for f in body_ids]]
        body_dof_vel = qvel[[f+6 for f in body_ids]]
        wrist_dof_pos = qpos[[f+7 for f in wrist_ids]]
        wrist_dof_vel = qvel[[f+6 for f in wrist_ids]]

        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        return whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel

    def _apply_global_recovery(self, action_mimic, robot_yaw, t_sim):
        """Bias the mimic velocity command with Kp feedback on the global pose error.

        Reads the reference planar pose (motion-file frame) published by the motion
        server, anchors it to the sim world on first sight (so robot and reference do
        not need to share coordinates), then injects a clipped corrective velocity into
        mimic dims [4:6] (ref-local xy vel) and dim [7] (yaw rate). mimic layout:
        [0]=height, [1:4]=rpy, [4:7]=root vel, [7]=yaw rate, [8:33]=dof.

        Also called with gains at 0 when --log_recovery or --push_force is set: the
        pose error is computed and logged, but the command is left untouched.
        """
        ref_pose_json = self.redis_client.get("ref_root_pose_g1")
        if ref_pose_json is None:
            self.recovery_anchor = None  # motion server idle/restarted -> re-anchor later
            self.recovery_prev_err = None
            self.recovery_err_dot = np.zeros(3)
            return action_mimic

        ref_x, ref_y, ref_yaw = json.loads(ref_pose_json)
        ref_xy = np.array([ref_x, ref_y])
        robot_xy = self.data.qpos[:2].copy()

        if self.recovery_anchor is None:
            dyaw = wrap_to_pi(robot_yaw - ref_yaw)
            self.recovery_anchor = (ref_xy.copy(), robot_xy.copy(), dyaw)
            self.motion_start_t = t_sim
            print(f"[Recovery] anchored: dyaw={dyaw:.3f}, ref0={ref_xy}, robot0={robot_xy}")

        ref_xy0, robot_xy0, dyaw = self.recovery_anchor
        c, s = np.cos(dyaw), np.sin(dyaw)
        rot = np.array([[c, -s], [s, c]])
        target_xy = robot_xy0 + rot @ (ref_xy - ref_xy0)
        target_yaw = ref_yaw + dyaw

        err_xy = target_xy - robot_xy
        yaw_err = wrap_to_pi(target_yaw - robot_yaw)

        # D term: finite-difference the error between control steps, EMA-filtered
        if self.recovery_prev_err is not None:
            prev_t, prev_err_xy, prev_yaw_err = self.recovery_prev_err
            dt = t_sim - prev_t
            if dt > 0:
                raw_dot = np.array([(err_xy[0] - prev_err_xy[0]) / dt,
                                    (err_xy[1] - prev_err_xy[1]) / dt,
                                    wrap_to_pi(yaw_err - prev_yaw_err) / dt])
                self.recovery_err_dot = 0.5 * raw_dot + 0.5 * self.recovery_err_dot
        self.recovery_prev_err = (t_sim, err_xy.copy(), yaw_err)

        vel_corr_world = np.clip(self.kp_recovery * err_xy
                                 + self.kd_recovery * self.recovery_err_dot[:2],
                                 -self.max_recovery_vel, self.max_recovery_vel)
        # express in the reference-local frame, matching the mimic obs convention
        ct, st = np.cos(target_yaw), np.sin(target_yaw)
        vel_corr_local = np.array([ct * vel_corr_world[0] + st * vel_corr_world[1],
                                   -st * vel_corr_world[0] + ct * vel_corr_world[1]])
        yaw_rate_corr = np.clip(self.kp_yaw_recovery * yaw_err
                                + self.kd_yaw_recovery * self.recovery_err_dot[2],
                                -self.max_recovery_yaw_rate, self.max_recovery_yaw_rate)

        if self.recovery_log is not None:
            self.recovery_log.append([t_sim - self.motion_start_t,
                                      target_xy[0], target_xy[1], robot_xy[0], robot_xy[1],
                                      target_yaw, robot_yaw,
                                      vel_corr_local[0], vel_corr_local[1], yaw_rate_corr])

        if self.recovery_active:
            action_mimic = action_mimic.copy()
            action_mimic[4:6] += vel_corr_local
            action_mimic[7] += yaw_rate_corr
        return action_mimic

    def _apply_push(self, t_sim):
        """Apply the scripted horizontal push to the pelvis inside its time window."""
        if self.motion_start_t is None:
            return
        t_motion = t_sim - self.motion_start_t
        if self.push_time <= t_motion < self.push_time + self.push_duration:
            self.data.xfrc_applied[self.pelvis_body_id, 0] = self.push_force * np.cos(self.push_angle)
            self.data.xfrc_applied[self.pelvis_body_id, 1] = self.push_force * np.sin(self.push_angle)
        else:
            self.data.xfrc_applied[self.pelvis_body_id, :3] = 0.0

    def _save_recovery_log(self):
        if self.recovery_log is None:
            return
        log_arr = np.array(self.recovery_log, dtype=np.float64).reshape(-1, 10)
        log_path = os.path.abspath(self.log_recovery)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        np.savez(log_path,
                 data=log_arr,
                 columns=np.array(["t", "target_x", "target_y", "robot_x", "robot_y",
                                   "target_yaw", "robot_yaw",
                                   "vel_corr_x", "vel_corr_y", "yaw_rate_corr"]),
                 kp_recovery=self.kp_recovery,
                 kp_yaw_recovery=self.kp_yaw_recovery,
                 kd_recovery=self.kd_recovery,
                 kd_yaw_recovery=self.kd_yaw_recovery,
                 max_recovery_vel=self.max_recovery_vel,
                 max_recovery_yaw_rate=self.max_recovery_yaw_rate,
                 push_force=self.push_force,
                 push_time=self.push_time,
                 push_duration=self.push_duration,
                 push_angle_deg=np.rad2deg(self.push_angle))
        print(f"[Recovery] tracking log saved to {log_path} ({log_arr.shape[0]} steps)")

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, mujoco_dof_pos=None):
        # body & hand
        self.data.qpos[:] = mujoco_dof_pos
        mujoco.mj_forward(self.model, self.data)
       
    def run(self):
        # Optionally record video (offscreen renderer: the passive viewer has no
        # read_pixels in mujoco >= 3)
        if self.record_video:
            import imageio
            print(f"Saving video to {self.video_path}")
            mp4_writer = imageio.get_writer(self.video_path, fps=50)
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 720)
            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            record_cam = mujoco.MjvCamera()
            record_cam.distance = 2.5
            record_cam.elevation = -15
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")

        # send initial proprio to redis
        proprio_json = json.dumps(self.proprio_history_buf[0].tolist())
        self.redis_client.set("state_body_g1", proprio_json)
        self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))
        try:
            for i in pbar:
                
                t_start = time.time()
                whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    
                    # Build a "proprio" vector for your policy, e.g.:
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = body_dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2],
                        (body_dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])
                    # send proprio to redis
                    self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))
                    self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))

                    # Try to get the latest mimic obs from Redis
                    try:
                        action_mimic_json = self.redis_client.get("action_mimic_g1")
                        if action_mimic_json is not None:
                            action_mimic_list = json.loads(action_mimic_json)
                            action_mimic = np.array(action_mimic_list, dtype=np.float32)
                            if self.recovery_active or self.recovery_log is not None or self.push_force != 0.0:
                                action_mimic = self._apply_global_recovery(action_mimic, rpy[2], i * self.sim_dt)
                            action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(action_mimic)
                        else:
                            raise Exception("cannot get action mimic from redis")
                    except:
                        raise Exception("cannot get action mimic from redis")

                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    obs_buf = np.concatenate([obs_full, obs_hist])
                    self.proprio_history_buf.append(obs_full)

                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    pd_target = aggregate_wrist_dof_pos(pd_target, wrist_dof_pos)
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    if self.viewer is not None:
                        # debug draw velocity arrow if you want
                        self.viewer.user_scn.ngeom = 0
                        draw_root_velocity(self.model, self.data, self.viewer, [0,0,0], 0, "pelvis", [1,0,0,1])
                        # make camera follow the pelvis
                        self.viewer.cam.lookat = pelvis_pos
                        self.viewer.sync()
                    if mp4_writer is not None:
                        record_cam.lookat = pelvis_pos
                        renderer.update_scene(self.data, camera=record_cam)
                        mp4_writer.append_data(renderer.render())

                # PD control
                torque = (pd_target - whole_body_dof) * self.stiffness - whole_body_dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                self.data.ctrl[:] = torque

                if self.push_force != 0.0:
                    self._apply_push(i * self.sim_dt)
                mujoco.mj_step(self.model, self.data)
                # sleep to maintain (scaled) real-time pace
                elapsed = time.time() - t_start
                if elapsed < self.wall_dt:
                    time.sleep(self.wall_dt - elapsed)
        except Exception as e:
            print(f"Error in run: {e}")
            pass
        finally:
            # persist data before any GL teardown: a fatal X error in
            # renderer/viewer close kills the process without unwinding Python
            self._save_recovery_log()
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")
                renderer.close()

            if self.viewer is not None:
                self.viewer.close()


def main_low_level_sim(args):
    controller = RealTimePolicyController(
        xml_file=args.xml_file,
        policy_path=args.policy_path,
        device='cuda',
        record_video=args.record_video,
        video_path=args.video_path,
        headless=args.headless,
        sim_duration=args.sim_duration,
        real_time_factor=args.real_time_factor,
        kp_recovery=args.kp_recovery,
        kp_yaw_recovery=args.kp_yaw_recovery,
        kd_recovery=args.kd_recovery,
        kd_yaw_recovery=args.kd_yaw_recovery,
        max_recovery_vel=args.max_recovery_vel,
        max_recovery_yaw_rate=args.max_recovery_yaw_rate,
        log_recovery=args.log_recovery,
        push_force=args.push_force,
        push_time=args.push_time,
        push_duration=args.push_duration,
        push_angle_deg=args.push_angle,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--xml_file", default=os.path.join(HERE, "../assets/g1/g1_sim2sim_with_wrist_roll.xml"), help="Mujoco XML file")
    
    parser.add_argument("--policy_path",  help="Path to the policy",
                        default="../assets/twist_general_motion_tracker.pt"
                        )
                        
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    parser.add_argument("--video_path", type=str, default="debug_sim.mp4",
                        help="output path for --record_video")
    parser.add_argument("--headless", action="store_true",
                        help="run without the interactive viewer window (scripted evals)")
    parser.add_argument("--sim_duration", type=float, default=100000.0,
                        help="stop the sim after this many sim-seconds (finite value = clean exit for scripted evals)")
    parser.add_argument("--real_time_factor", type=float, default=1.0,
                        help="pace the sim at this fraction of real time; pair with the same factor on the motion server when the sim cannot sustain 1.0")

    # extension: outer-loop global-trajectory recovery (0 = off, i.e. original behavior)
    parser.add_argument("--kp_recovery", type=float, default=0.0,
                        help="Kp gain on global xy position error, injected via the root velocity command")
    parser.add_argument("--kp_yaw_recovery", type=float, default=0.0,
                        help="Kp gain on heading error, injected via the yaw-rate command")
    parser.add_argument("--kd_recovery", type=float, default=0.0,
                        help="Kd gain on the xy error rate (damping for the position correction)")
    parser.add_argument("--kd_yaw_recovery", type=float, default=0.0,
                        help="Kd gain on the heading error rate (damping for the yaw correction)")
    parser.add_argument("--max_recovery_vel", type=float, default=0.5,
                        help="cap [m/s] on the corrective linear velocity")
    parser.add_argument("--max_recovery_yaw_rate", type=float, default=1.0,
                        help="cap [rad/s] on the corrective yaw rate")

    # extension: evaluation tooling for A/B comparison (see data_utils/compare_recovery.py).
    # Both need the patched motion server (it publishes ref_root_pose_g1); run one motion
    # per sim run so the log has a single time axis.
    parser.add_argument("--log_recovery", type=str, default=None,
                        help="save a per-step global-tracking log to this .npz; also works with gains at 0 to record a baseline")
    parser.add_argument("--push_force", type=float, default=0.0,
                        help="scripted horizontal push on the pelvis [N] to induce drift (0 = no push)")
    parser.add_argument("--push_time", type=float, default=2.0,
                        help="push onset [s after the motion reference starts]")
    parser.add_argument("--push_duration", type=float, default=0.2,
                        help="push duration [s]")
    parser.add_argument("--push_angle", type=float, default=90.0,
                        help="push direction in the world xy plane [deg, 0 = +x, 90 = +y]")
    args = parser.parse_args()

    args.record_proprio = True
    
    main_low_level_sim(args)
