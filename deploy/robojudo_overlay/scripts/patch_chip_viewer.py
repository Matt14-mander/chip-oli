"""Idempotent MuJoCo viewer camera and body-selection compatibility patch."""

import ast
from pathlib import Path

import mujoco


def patch(path):
    # Older MuJoCo uses the scene argument and must retain it.
    old_camera_api = "scn:" in (mujoco.mjv_moveCamera.__doc__ or "")
    raw = path.read_bytes()
    src = raw.decode("utf-8")
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(ast.parse(src)):
        if (not old_camera_api
                and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "mjv_moveCamera" and len(node.args) == 6
                and not node.keywords):
            args = node.args[:4] + node.args[5:]
            replacement = ast.get_source_segment(src, node.func) + "(" + ", ".join(
                ast.get_source_segment(src, arg) for arg in args) + ")"
            edits.append((offsets[node.lineno - 1] + node.col_offset,
                          offsets[node.end_lineno - 1] + node.end_col_offset,
                          replacement.encode("utf-8")))
        # mjv_select writes a one-element NumPy output array. The perturbation
        # setter requires a scalar; accept Python/NumPy scalars as well.
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "skinselect"
                and isinstance(node.value, ast.Name) and node.value.id == "selskin"):
            value = node.value
            replacement = 'int(selskin.item() if hasattr(selskin, "item") else selskin)'
            edits.append((offsets[value.lineno - 1] + value.col_offset,
                          offsets[value.end_lineno - 1] + value.end_col_offset,
                          replacement.encode("utf-8")))
    if edits:
        backup = path.with_suffix(".py.before-chip-camera-fix")
        if not backup.exists():
            backup.write_bytes(raw)
        for start, end, replacement in sorted(edits, reverse=True):
            raw = raw[:start] + replacement + raw[end:]
        ast.parse(raw.decode("utf-8"))
        path.write_bytes(raw)
    print(f"Viewer compatibility patch: {len(edits)} edit(s) applied")


if __name__ == "__main__":
    patch(Path(__file__).resolve().parents[1] /
          "third_party/mujoco_viewer/mujoco_viewer/callbacks.py")
