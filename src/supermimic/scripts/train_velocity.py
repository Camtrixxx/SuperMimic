"""Train a velocity-command encoder on a frozen, trained motion tracker.

Example:
  CUDA_VISIBLE_DEVICES=4 uv run sm-train-velocity \
    --init-run-dir logs/SuperMimic-GenCoord-G1/<motion run> --init-snapshot it0112000.pt \
    --mode rl --motion-file data/loco/packed --run-name vel_rl
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.runners.velocity_runner import (
  VelocityRunner,
  VelocityRunnerCfg,
  load_base_cfg,
)
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.velocity import VelocityCfg, fall_termination


@dataclass
class Args:
  init_run_dir: str
  init_snapshot: str
  motion_file: str
  run_name: str
  mode: str = "rl"
  task_id: str = "SuperMimic-GenCoord-G1"
  num_envs: int = 4096
  max_iterations: int = 100_001
  save_interval: int = 10_000
  seed: int = 0
  clean_action_frac: float = 0.5
  lr_schedule: str = "cosine"
  device: str | None = None


def main(a: Args) -> None:
  assert a.mode in ("rl", "distill"), a.mode
  device = setup_device(a.device)
  base = load_base_cfg(a.init_run_dir)
  cfg = VelocityRunnerCfg(
    **{f.name: getattr(base, f.name) for f in dataclasses.fields(SuperMimicRunnerCfg)},
    velocity=VelocityCfg(
      mode=a.mode, init_run_dir=a.init_run_dir, init_snapshot=a.init_snapshot
    ),
  )
  cfg.run_name = a.run_name
  cfg.max_iterations = a.max_iterations
  cfg.save_interval = a.save_interval
  cfg.sampling_mode = "uniform"
  cfg.clean_action_frac = a.clean_action_frac
  cfg.lr_schedule = a.lr_schedule  # type: ignore[assignment]
  cfg.rank, cfg.world_size = 0, 1

  env_cfg = load_env_cfg(a.task_id, play=False)
  env_cfg.seed = a.seed
  env_cfg.commands["motion"].motion_file = a.motion_file
  env_cfg.commands["motion"].sampling_mode = "uniform"
  if a.mode == "rl":
    # Fall-only termination in sim (Table A.6). The tracking terms stay in the
    # cfg (the WM env reads their thresholds) but can never fire.
    for name in ("anchor_pos", "anchor_ori"):
      env_cfg.terminations[name].params["threshold"] = 1e6
    env_cfg.terminations["fall"] = TerminationTermCfg(func=fall_termination)
  env = create_env(env_cfg, a.num_envs, device)
  VelocityRunner(env=env, cfg=cfg, task_id=a.task_id, device=device).run()


def cli() -> None:
  """Console entry point ``sm-train-velocity``."""
  main(tyro.cli(Args))


if __name__ == "__main__":
  cli()
