"""OLI HU_D04 gripper model shared with CHIP simulation deployment."""

import json
import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def robot_metadata():
  return json.loads((ASSET_DIR / "robot.json").read_text(encoding="utf-8"))


def get_spec():
  spec = mujoco.MjSpec.from_file(os.path.relpath(ASSET_DIR / "robot.xml"))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


def get_oli_robot_cfg():
  meta = robot_metadata()
  actuators = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,), stiffness=p["stiffness"], damping=p["damping"],
      effort_limit=p["effort_limit"], armature=p["armature"],
    )
    for name, p in meta["joints"].items()
  )
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=tuple(meta["default_base_position"]),
      joint_pos={n: p["default_position"] for n, p in meta["joints"].items()},
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    collisions=(CollisionCfg(
      geom_names_expr=(".*_collision_.*",), contype=0, conaffinity=1,
      condim={".*_ankle_roll_link_collision_.*": 3, ".*_collision_.*": 1},
      priority=1, friction=(0.6,),
    ),),
  )


def action_scale():
  return {n: p["action_scale"] for n, p in robot_metadata()["joints"].items()}
