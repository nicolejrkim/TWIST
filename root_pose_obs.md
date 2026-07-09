# Extension: Reference Root Pose Observation (training-side)

Branch `extension/train-root-pose-obs`. Trains the drift correction into the policy that
`extension/deploy-global-recovery` bolts on at deploy time: the stock TWIST policy never
observes where the reference root *is* — only its height, orientation and velocities — so
global position error is unobservable and drift (especially heading, on turning motions)
grows unbounded. This extension feeds the policy the reference root pose the same way it
already receives the reference root height, and rewards closing the error.

## What the policy sees

3 dims are appended to **each mimic obs block** (teacher: every target step; student: the
single target step): `[err_x, err_y, err_yaw]` — the reference root pose relative to the
robot's current root, expressed in the robot's heading (yaw) frame:

- `err_xy` = (ref root xy − robot root xy) rotated by −yaw_robot, clipped to
  `±env.ref_root_pose_err_clip` (default 1.0 m)
- `err_yaw` = wrap(ref yaw − robot yaw) to [−π, π]

Relative-error form, not absolute (unlike the height): the robot has no observation of its
own world xy/yaw, so an absolute reference would be uninformative. The error form is exactly
the signal the deploy-time recovery controller computes from odometry.

## Dimension changes (G1)

| quantity | old | new |
|---|---|---|
| `n_mimic_obs` (student, per frame) | 31 = 8 + 23 | 34 = 8 + 23 + 3 |
| teacher per-step block | 58 = 8 + 23 + 27 | 61 |
| `n_priv_mimic_obs` (20 steps) | 1160 | 1220 |
| student `num_observations` | 1155 | 1188 |
| teacher/priv `num_observations` | 1318 | 1378 |

The 3 dims are appended at the **end** of each block, so all legacy obs indices (and the
deploy-side mimic slicing, e.g. `extract_mimic_obs_to_body_and_wrist`, recovery dims
`[4:6]`/`[7]`) are unchanged.

## Training-side changes

- `HumanoidMimicCfg.env.obs_ref_root_pose` (default `False`; `True` in the G1 configs) —
  gates the obs, all dim arithmetic is conditional on it, so flipping it off reproduces the
  stock setup exactly for A/B training.
- `_reward_tracking_root_pose` includes the global xy error when the flag is on (before,
  the non-global branch tracked z only; yaw was already covered by the quat term).
- `env.randomize_init_root_pose_err` (on with the flag): at reset the robot is offset from
  the reference by ±`init_root_xy_err_range` (0.15 m) and ±`init_root_yaw_err_range`
  (0.2 rad), without moving `episode_init_origin`, so episodes start with a genuine pose
  error to close. Otherwise the initial error is exactly zero and errors only arise from
  pushes and drift.

Train as usual (`train_teacher.sh` → `train_student.sh`); both teacher and student get the
new obs so DAgger distills the recovery behavior. Export with
`save_jit_stu_rlbc.py --root_pose_obs` (wired into `to_jit.sh`; drop the flag for
checkpoints trained without the feature).

## Deploy contract (not implemented here)

The low-level server must append `[err_x, err_y, err_yaw]` to the 31-dim mimic obs before
assembling the policy input:

1. anchor the SE(2) transform motion-file frame → world on the first reference pose
   (`server_low_level_g1_sim.py` already does this in `_apply_global_recovery`, and the
   motion server already publishes `ref_root_pose_{robot}` = [x, y, yaw] to Redis);
2. compute the world error to the robot's odometry pose, rotate into the robot heading
   frame, clip xy to the same `ref_root_pose_err_clip` used in training;
3. append to the mimic obs; feed the policy. Outer-loop recovery gains stay at 0.
