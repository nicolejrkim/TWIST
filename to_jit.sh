# bash to_jit.sh 0927_twist_rlbcstu

cd legged_gym/legged_gym/scripts


exptid=${1}

proj_name="g1_stu_rl"

# Run the training script
# --root_pose_obs: policies trained on this branch observe the reference root pose
# (drop the flag when exporting checkpoints trained without env.obs_ref_root_pose)
python save_jit_stu_rlbc.py --robot "g1" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --checkpoint -1 \
                --root_pose_obs \