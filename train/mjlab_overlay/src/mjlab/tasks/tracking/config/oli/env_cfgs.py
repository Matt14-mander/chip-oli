"""CHIP task for the OLI HU_D04 with fixed grippers and 31 actuated joints."""

from mjlab.asset_zoo.robots.oli import action_scale, get_oli_robot_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_chip_tracking_env_cfg

OLI_TRACKING_BODY_NAMES = (
  "base_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "waist_pitch_link", "left_shoulder_roll_link", "left_elbow_link",
  "left_wrist_roll_link", "right_shoulder_roll_link", "right_elbow_link",
  "right_wrist_roll_link", "head_pitch_link",
)
OLI_CHIP_EEF_BODY_NAMES = (
  "waist_pitch_link", "left_wrist_roll_link", "right_wrist_roll_link",
)
OLI_ALLOWED_CONTACT_BODY_NAMES = (
  "left_ankle_roll_link", "right_ankle_roll_link",
  "left_wrist_roll_link", "right_wrist_roll_link",
)
# Keep the original CHIP command semantics: leg and waist reference positions/velocities.
OLI_COMMAND_JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
)


def oli_flat_chip_tracking_env_cfg(play=False):
  cfg = unitree_g1_flat_chip_tracking_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_oli_robot_cfg()}
  cfg.actions["joint_pos"].scale = action_scale()
  cfg.commands["motion"].anchor_body_name = "waist_pitch_link"
  cfg.commands["motion"].body_names = OLI_TRACKING_BODY_NAMES
  cfg.commands["motion"].eef_body_names = OLI_CHIP_EEF_BODY_NAMES
  cfg.commands["motion"].command_joint_names = OLI_COMMAND_JOINT_NAMES
  cfg.commands["motion"].motion_robot = "oli"
  cfg.commands["motion"].require_motion_metadata = True
  # The initial OLI model disables self collision, matching the ADAPT migration.
  cfg.scene.sensors = (ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
  ), ContactSensorCfg(
    name="undesired_contact",
    primary=ContactMatch(mode="subtree", pattern=r".*", entity="robot",
                         exclude=OLI_ALLOWED_CONTACT_BODY_NAMES),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
  ))
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"left_ankle_roll_link_collision_.*", r"right_ankle_roll_link_collision_.*",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_pitch_link",)
  cfg.viewer.body_name = "waist_pitch_link"
  for name in ("motion_3pt_pos_rel", "motion_3pt_ori_rel"):
    cfg.rewards[name].params["body_names"] = OLI_CHIP_EEF_BODY_NAMES
  if not play:
    cfg.events["chip_eef_impulse"].params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=OLI_CHIP_EEF_BODY_NAMES, preserve_order=True,
    )
  return cfg
