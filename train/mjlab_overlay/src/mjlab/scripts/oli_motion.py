"""Generate a local OLI motion NPZ using the training model's forward kinematics.

Standing mode is a pipeline check, not a retargeted walking reference.
Real CSV input must already be retargeted to OLI: XYZ + quaternion XYZW + 31 joints.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from mjlab.chip_common.motion import validate_motion
from mjlab.asset_zoo.robots.oli import get_oli_robot_cfg, robot_metadata
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.oli.env_cfgs import oli_flat_chip_tracking_env_cfg


def convert(output, input_file=None, joint_names_file=None, standing_seconds=None,
            input_fps=30, output_fps=50, device="cuda:0"):
  output = Path(output)
  if output.suffix != ".npz":
    raise ValueError("--output must end in .npz")
  if output.exists():
    raise FileExistsError(f"Choose a new output file: {output}")
  meta = robot_metadata()
  names = list(meta["joints"])
  if input_fps <= 0 or output_fps != 50:
    raise ValueError("Input fps must be positive; CHIP OLI currently uses 50 Hz")
  temporary = None
  if standing_seconds is not None:
    if standing_seconds <= 0:
      raise ValueError("standing_seconds must be positive")
    input_fps = output_fps
    frames = max(4, int(np.ceil(standing_seconds * input_fps)) + 1)
    pose = [*meta["default_base_position"], 0, 0, 0, 1,
            *[meta["joints"][n]["default_position"] for n in names]]
    temporary = tempfile.TemporaryDirectory(prefix="chip_oli_")
    input_file = Path(temporary.name) / "standing.csv"
    np.savetxt(input_file, np.tile(pose, (frames, 1)), delimiter=",")
  elif input_file is None or joint_names_file is None:
    raise ValueError("CSV conversion requires --input-file and --joint-names-file")
  else:
    order = json.loads(Path(joint_names_file).read_text(encoding="utf-8"))
    if isinstance(order, dict):
      order = list(order["joints"])
    if order != names:
      raise ValueError("CSV joint order must match OLI robot.json; retarget/reorder first")
  csv = np.loadtxt(input_file, delimiter=",", ndmin=2)
  if csv.shape[0] < 4 or csv.shape[1] != 38 or not np.isfinite(csv).all():
    raise ValueError("OLI CSV requires >=4 rows and 38 finite columns; G1 CSV is incompatible")
  if not np.allclose(np.linalg.norm(csv[:, 3:7], axis=1), 1, atol=1e-3):
    raise ValueError("CSV base quaternion must be unit XYZW")
  cfg = oli_flat_chip_tracking_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.scene.entities = {"robot": get_oli_robot_cfg()}
  scene = Scene(cfg.scene, device=device)
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  # A fixed model convention is required by both the loader and RoboJuDo.
  if list(robot.joint_names) != names:
    raise ValueError("Compiled OLI joint order differs from robot.json")
  scene.reset()
  motion = MotionLoader(str(input_file), input_fps, output_fps, device)
  keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
          "body_lin_vel_w", "body_ang_vel_w")
  log = {key: [] for key in keys}
  for _ in range(motion.output_frames):
    (pos, quat, lin_vel, ang_vel, joint_pos, joint_vel), _ = motion.get_next_state()
    root = robot.data.default_root_state.clone()
    root[:, :3] = pos
    root[:, 3:7] = quat
    root[:, 7:10] = lin_vel
    root[:, 10:] = ang_vel
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    arrays = (robot.data.joint_pos, robot.data.joint_vel, robot.data.body_link_pos_w,
              robot.data.body_link_quat_w, robot.data.body_link_lin_vel_w,
              robot.data.body_link_ang_vel_w)
    for key, array in zip(keys, arrays, strict=True):
      log[key].append(array[0].cpu().numpy().copy())
  log = {key: np.stack(value) for key, value in log.items()}
  log.update(fps=np.array([output_fps]), robot=np.array("oli"),
             joint_names=np.array(names), body_names=np.array(robot.body_names))
  validate_motion(log, names, robot.body_names, robot="oli", require_metadata=True)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **log)
  if temporary is not None:
    temporary.cleanup()
  print(f"Saved {output.resolve()}: {len(log['joint_pos'])} frames, 31 joints, "
        f"{len(robot.body_names)} bodies at {output_fps} Hz")
  return output


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--input-file", type=Path)
  source.add_argument("--standing-seconds", type=float)
  parser.add_argument("--joint-names-file", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--input-fps", type=float, default=30)
  parser.add_argument("--output-fps", type=float, default=50)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  convert(**vars(args))


if __name__ == "__main__":
  main()
