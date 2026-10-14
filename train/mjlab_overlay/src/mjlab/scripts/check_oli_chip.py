"""Step the OLI CHIP training and play environments against a named motion NPZ."""

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  # Import registers the OLI task, also when launched via python -m.
  import mjlab.tasks  # noqa: F401
  task = "Mjlab-Tracking-Flat-LimX-OLI-HU-D04-CHIP"
  for play in (False, True):
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = 2
    cfg.commands["motion"].motion_file = args.motion_file
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
      env.reset()
      for _ in range(10):
        obs, rewards, _, _, _ = env.step(torch.zeros((2, 31), device=args.device))
        assert obs["actor"].shape == (2, 1059), obs["actor"].shape
        if not torch.isfinite(obs["actor"]).all() or not torch.isfinite(rewards).all():
          raise RuntimeError("Non-finite OLI CHIP environment state")
      print(f"PASS {task} play={play}: actor={tuple(obs['actor'].shape)}, actions=31")
    finally:
      env.close()


if __name__ == "__main__":
  main()
