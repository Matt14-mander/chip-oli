"""Simulation-only OLI CHIP pipeline, using the same assets and gains as mjlab."""

import json
import os
from pathlib import Path

import mujoco
from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy.policy_cfgs import ChipTrackingPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "oli"
META = json.loads((ASSETS / "robot.json").read_text(encoding="utf-8"))
JOINTS = META["joints"]
MODEL = mujoco.MjModel.from_xml_path(os.path.relpath(ASSETS / "scene.xml"))
MODEL_JOINTS = [MODEL.joint(i).name for i in range(1, MODEL.njnt)]
if MODEL_JOINTS != list(JOINTS) or (MODEL.nq, MODEL.nv, MODEL.nu) != (38, 37, 31):
    raise ValueError("OLI model and joint metadata do not agree")
BODY_NAMES = [MODEL.body(i).name for i in range(1, MODEL.nbody)]


def oli_dof():
    return DoFConfig(
        joint_names=list(JOINTS),
        default_pos=[p["default_position"] for p in JOINTS.values()],
        stiffness=[p["stiffness"] for p in JOINTS.values()],
        damping=[p["damping"] for p in JOINTS.values()],
        torque_limits=[p["effort_limit"] for p in JOINTS.values()],
        position_limits=[MODEL.jnt_range[i].tolist() for i in range(1, MODEL.njnt)],
    )


@cfg_registry.register
class oli_chip(RlPipelineCfg):
    robot: str = "oli"
    env: MujocoEnvCfg = MujocoEnvCfg(
        xml=str(ASSETS / "scene.xml"), dof=oli_dof(),
        torso_name=META["torso_body"], imu_site_name="imu",
        initial_base_position=META["default_base_position"],
        born_place_align=False, random_heading=False,
        sim_dt=0.005, sim_decimation=4, sim_iterations=10, sim_ls_iterations=20,
        actuator_mode="position_servo",
        actuator_kp_override={n: p["stiffness"] for n, p in JOINTS.items()},
        actuator_kd_override={n: p["damping"] for n, p in JOINTS.items()},
        actuator_effort_override={n: p["effort_limit"] for n, p in JOINTS.items()},
        joint_armature_override={n: p["armature"] for n, p in JOINTS.items()},
        joint_damping_override={n: 0.0 for n in JOINTS},
        joint_frictionloss_override={n: 0.0 for n in JOINTS},
    )
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg(triggers={
        "r": "[MOTION_RESET]", "i": "[SIM_REBORN]", "o": "[SHUTDOWN]",
        "<": "[MOTION_FADE_IN]", ">": "[MOTION_FADE_OUT]",
    })]
    policy: ChipTrackingPolicyCfg = ChipTrackingPolicyCfg(
        robot="oli", obs_dof=oli_dof(), action_dof=oli_dof(),
        root_body_name=META["base_body"], anchor_body_name=META["torso_body"],
        keypoint_body_names=[META["torso_body"], *META["hand_bodies"]],
        command_joint_names=list(JOINTS)[:15],
        motion_body_names=BODY_NAMES, require_motion_metadata=True,
        joint_action_scales=[p["action_scale"] for p in JOINTS.values()],
        motion_time_end=None, compliance_range=[0.0, 0.15],
    )
