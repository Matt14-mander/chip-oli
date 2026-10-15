"""Exercise the viewer patch without opening a GLFW window."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize("old_camera_api", [False, True])
def test_viewer_camera_and_selection_patch(tmp_path, old_camera_api):
    script = (Path(__file__).resolve().parents[1] /
              "deploy/robojudo_overlay/scripts/patch_chip_viewer.py")
    spec = importlib.util.spec_from_file_location("viewer_patch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def camera():
        pass

    camera.__doc__ = "scn: MjvScene" if old_camera_api else "model: MjModel"
    module.mujoco = SimpleNamespace(mjv_moveCamera=camera)
    path = tmp_path / "callbacks.py"
    original = (
        '# 鼠标选择\r\n'
        'def select(self, selskin):\r\n'
        '    self.pert.skinselect = selskin\r\n'
        'def move(self):\r\n'
        '    mujoco.mjv_moveCamera(self.model, action, dx, dy, self.scn, self.cam)\r\n'
    ).encode("utf-8")
    path.write_bytes(original)
    module.patch(path)
    patched = path.read_bytes()
    tree = ast.parse(patched)
    call = tree.body[1].body[0].value
    assert len(call.args) == (6 if old_camera_api else 5)
    assert path.with_suffix(".py.before-chip-camera-fix").read_bytes() == original

    namespace = {}
    exec(compile(patched, str(path), "exec"), namespace)
    viewer = SimpleNamespace(pert=mujoco.MjvPerturb())
    for selection in (np.array([[-1]], dtype=np.int32), np.array([3]), np.int32(2), 1):
        namespace["select"](viewer, selection)
        assert viewer.pert.skinselect == int(np.asarray(selection).item())

    module.patch(path)
    assert path.read_bytes() == patched
    assert path.with_suffix(".py.before-chip-camera-fix").read_bytes() == original
