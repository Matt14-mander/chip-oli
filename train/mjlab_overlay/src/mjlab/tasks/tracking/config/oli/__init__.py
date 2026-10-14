from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import oli_flat_chip_tracking_env_cfg
from .rl_cfg import oli_chip_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-LimX-OLI-HU-D04-CHIP",
  env_cfg=oli_flat_chip_tracking_env_cfg(),
  play_env_cfg=oli_flat_chip_tracking_env_cfg(play=True),
  rl_cfg=oli_chip_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
