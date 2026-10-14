"""Local CPU checks; no upstream checkout or GPU is required.

The actual attention class definitions are loaded in isolation so these checks
can also run where mjlab/RoboJuDo's optional runtime dependencies are absent.
"""

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from chip_common.motion import validate_motion

META = json.loads((ROOT / "robot_assets/oli/robot.json").read_text(encoding="utf-8"))
TRAIN = ROOT / "train/mjlab_overlay/src/mjlab"
DEPLOY = ROOT / "deploy/robojudo_overlay"


def model():
    # Relative paths support MuJoCo's Windows Unicode-path limitation.
    import os
    return mujoco.MjModel.from_xml_path(os.path.relpath(ROOT / "robot_assets/oli/scene.xml"))


def attention_classes():
    def definitions(path, wanted, namespace):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
        exec(compile(tree, str(path), "exec"), namespace)
        return namespace

    ns = {"torch": torch, "nn": nn, "math": math, "Any": Any,
          "resolve_nn_activation": lambda name: {"elu": nn.ELU}[name]()}
    training = definitions(TRAIN / "rl/chip_attention_model.py", {
        "_activation", "_attention_defaults", "_merge_attention_cfg",
        "SinusoidalPositionalEncoding", "ChipTemporalCrossAttentionCore",
    }, ns)
    deployment = definitions(DEPLOY / "robojudo/policy/chip_tracking_policy.py", {
        "_SinusoidalPositionalEncoding", "_ChipTemporalCrossAttentionCore", "_ChipAttentionActor",
    }, {"torch": torch, "math": math})
    return training, deployment


def valid_motion():
    m = model()
    n = m.nbody - 1
    quats = np.zeros((4, n, 4))
    quats[..., 0] = 1
    return {
        "robot": np.array("oli"), "fps": np.array([50]),
        "joint_names": np.array(list(META["joints"])),
        "body_names": np.array([m.body(i).name for i in range(1, m.nbody)]),
        "joint_pos": np.zeros((4, 31)), "joint_vel": np.zeros((4, 31)),
        "body_pos_w": np.zeros((4, n, 3)), "body_quat_w": quats,
        "body_lin_vel_w": np.zeros((4, n, 3)), "body_ang_vel_w": np.zeros((4, n, 3)),
    }


def test_oli_model_and_pd_mapping():
    m = model()
    assert (m.nq, m.nv, m.nu) == (38, 37, 31)
    assert [m.joint(i).name for i in range(1, m.njnt)] == list(META["joints"])
    for i, (name, params) in enumerate(META["joints"].items()):
        joint = m.joint(name)
        assert m.actuator_trnid[i, 0] == joint.id
        assert m.actuator(i).name == name
        assert m.actuator_gainprm[i, 0] == pytest.approx(params["stiffness"])
        assert m.actuator_biasprm[i, 2] == pytest.approx(-params["damping"])
        assert m.dof_armature[m.jnt_dofadr[joint.id]] == pytest.approx(params["armature"], rel=1e-5)
        assert params["action_scale"] == pytest.approx(0.25 * params["effort_limit"] / params["stiffness"])
    d = mujoco.MjData(m)
    d.qpos[:3] = META["default_base_position"]
    d.qpos[7:] = [p["default_position"] for p in META["joints"].values()]
    d.ctrl[:] = d.qpos[7:]
    for _ in range(50):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()


def test_all_oli_tracking_bodies_exist():
    namespace = {}
    tree = ast.parse((TRAIN / "tasks/tracking/config/oli/env_cfgs.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.Assign)]
    exec(compile(tree, "oli_constants", "exec"), namespace)
    m = model()
    bodies = {m.body(i).name for i in range(1, m.nbody)}
    for key in ("OLI_TRACKING_BODY_NAMES", "OLI_CHIP_EEF_BODY_NAMES", "OLI_ALLOWED_CONTACT_BODY_NAMES"):
        assert set(namespace[key]) <= bodies
    assert set(namespace["OLI_COMMAND_JOINT_NAMES"]) <= set(META["joints"])


@pytest.mark.parametrize("joints,obs_dim", [(29, 999), (31, 1059)])
def test_attention_training_deployment_parity(joints, obs_dim):
    training, deployment = attention_classes()
    cfg = {"history_length": 10, "embed_dim": 128, "num_heads": 4,
           "num_layers": 1, "ff_dim": 256, "dropout": 0,
           "layout": {"joint_pos_dim": joints, "joint_vel_dim": joints, "actions_dim": joints}}
    train = training["ChipTemporalCrossAttentionCore"](obs_dim, cfg, "elu").eval()
    deploy = deployment["_ChipTemporalCrossAttentionCore"](
        obs_dim, 10, 128, 4, 1, 256, num_joints=joints).eval()
    deploy.load_state_dict(train.state_dict(), strict=True)
    obs = torch.randn(3, obs_dim)
    with torch.inference_mode():
        torch.testing.assert_close(train(obs), deploy(obs), atol=1e-6, rtol=1e-6)
    actor = deployment["_ChipAttentionActor"](obs_dim, (512, 256, 128), joints, 10, 128, 4, 1, 256)
    assert actor(obs).shape == (3, joints)


def test_named_motion_contract():
    data = valid_motion()
    validate_motion(data, list(META["joints"]), data["body_names"], "oli", True)


@pytest.mark.parametrize("problem", ["robot", "joint_order", "body_order", "missing_names", "nan", "quat", "fps", "g1"])
def test_incompatible_motion_rejected(problem):
    data = valid_motion()
    expected_bodies = data["body_names"].copy()
    if problem == "robot":
        data["robot"] = np.array("g1")
    elif problem == "joint_order":
        data["joint_names"] = data["joint_names"][::-1]
    elif problem == "body_order":
        data["body_names"] = data["body_names"][::-1]
    elif problem == "missing_names":
        del data["body_names"]
    elif problem == "nan":
        data["joint_vel"][0, 0] = np.nan
    elif problem == "quat":
        data["body_quat_w"][0, 0] = 0
    elif problem == "fps":
        data["fps"] = np.array([0])
    elif problem == "g1":
        data["joint_pos"] = np.zeros((4, 29))
    with pytest.raises(ValueError):
        validate_motion(data, list(META["joints"]), expected_bodies, "oli", True)
