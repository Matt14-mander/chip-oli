from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_chip_tracking_ppo_runner_cfg


def oli_chip_tracking_ppo_runner_cfg():
  cfg = unitree_g1_chip_tracking_ppo_runner_cfg()
  cfg.experiment_name = "oli_chip_tracking_attn"
  cfg.actor.attention_cfg["layout"] = {
    "joint_pos_dim": 31, "joint_vel_dim": 31, "actions_dim": 31,
  }
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg
