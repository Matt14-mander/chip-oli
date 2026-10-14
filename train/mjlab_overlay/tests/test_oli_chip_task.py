"""Robot-specific CHIP configuration regression checks."""

import pytest
from mjlab.asset_zoo.robots.oli import robot_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = "Mjlab-Tracking-Flat-LimX-OLI-HU-D04-CHIP"


@pytest.mark.parametrize("play", [False, True])
def test_oli_chip_configuration(play):
  cfg = load_env_cfg(TASK, play=play)
  motion = cfg.commands["motion"]
  assert motion.motion_robot == "oli" and motion.require_motion_metadata
  assert motion.anchor_body_name == "waist_pitch_link"
  assert motion.eef_body_names == (
    "waist_pitch_link", "left_wrist_roll_link", "right_wrist_roll_link",
  )
  assert len(motion.command_joint_names) == 15
  assert len(cfg.actions["joint_pos"].scale) == 31
  assert set(cfg.actions["joint_pos"].scale) == set(robot_metadata()["joints"])
  assert "chip_eef_force_b" not in cfg.observations["actor"].terms
  assert "chip_eef_force_b" in cfg.observations["critic"].terms
  if not play:
    assert cfg.events["chip_eef_impulse"].params["asset_cfg"].body_names == motion.eef_body_names
  else:
    assert "chip_eef_impulse" not in cfg.events


def test_oli_actor_layout_and_g1_compatibility():
  rl = load_rl_cfg(TASK)
  assert rl.actor.attention_cfg["layout"] == {
    "joint_pos_dim": 31, "joint_vel_dim": 31, "actions_dim": 31,
  }
  assert rl.experiment_name == "oli_chip_tracking_attn"
  g1 = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-CHIP")
  assert g1.commands["motion"].anchor_body_name == "torso_link"
  assert not g1.commands["motion"].require_motion_metadata
