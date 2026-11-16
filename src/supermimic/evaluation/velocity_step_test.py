"""Velocity step test (paper Appendix G.2).

From standing, each command is held for 5 s; 4 trials per command. Relative
steady-state error = || mean velocity error over the last 2 s || / || command ||
in (vx, vy, wz) space, robot velocity measured in its heading frame; the
per-command value is the median over trials that did not fall. Commands: a
15 x 15 grid on each of the planes (vx, vy), (vx, wz), (vy, wz) spanning the
uniform training range, third component zero, standing command left out ->
3 * 224 = 672 step responses.

Our choices where the paper is silent: the standing start is frame 0 of a
BONES-SEED standing-idle clip held static; the 4 trials differ by the training
sensor noise (friction 1.0, no other DR); a fall uses the Table A.6 rule.

Usage: uv run sm-velocity-step-test --run-dir <velocity run> [--snapshot ...]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from tensordict import TensorDict

from supermimic.evaluation.tracking import EvalConfig, Evaluator
from supermimic.paths import data_path, require
from supermimic.velocity import (
  CMD_HI,
  CMD_LO,
  VelocityActor,
  fall_error,
  measured_cmd_space,
)

DT, HOLD_S, TAIL_S, TRIALS, GRID = 0.02, 5.0, 2.0, 4, 15
PLANES = ((0, 1), (0, 2), (1, 2))


def command_grid() -> np.ndarray:
  cmds = []
  for a, b in PLANES:
    ga = np.linspace(CMD_LO[a], CMD_HI[a], GRID)
    gb = np.linspace(CMD_LO[b], CMD_HI[b], GRID)
    for x in ga:
      for y in gb:
        c = np.zeros(3)
        c[a], c[b] = x, y
        if np.linalg.norm(c) > 1e-6:
          cmds.append(c)
  return np.array(cmds, dtype=np.float32)


def make_standing(src: str, out: str, seconds: float = HOLD_S + 2.0) -> str:
  d = dict(np.load(src))
  n = int(seconds / DT)
  tp = {k: d[k] for k in ("fps", "joint_names")}
  for k in ("joint_pos", "root_pos_w", "root_quat_w"):
    tp[k] = np.repeat(d[k][:1], n, 0)
  for k, w in (
    ("joint_vel", d["joint_vel"].shape[1]),
    ("root_lin_vel_w", 3),
    ("root_ang_vel_w", 3),
  ):
    tp[k] = np.zeros((n, w), np.float32)
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **tp)
  return out


@dataclass
class Args:
  run_dir: str
  snapshot: str | None = None
  standing_src: str = ""
  """npz of a standing clip; default: first ``*idle*`` clip in <data>/loco/npz."""
  standing_out: str | None = None
  """Default <data>/derived/standing_static.npz."""
  output: str | None = None
  device: str | None = None


@torch.inference_mode()
def main(a: Args) -> None:
  if a.standing_src:
    src = a.standing_src
  else:
    idle = sorted(require(data_path("loco", "npz"), "loco library").glob("*idle*.npz"))
    if not idle:
      raise SystemExit("no *idle* clip in <data>/loco/npz; pass --standing-src")
    src = idle[0].as_posix()
  standing = make_standing(
    src, a.standing_out or str(data_path("derived", "standing_static.npz"))
  )
  cmds = torch.as_tensor(command_grid())
  n_cmd = cmds.shape[0]
  n = n_cmd * TRIALS
  cfg = EvalConfig(run_dir=a.run_dir, motion_file=standing, device=a.device)
  ev = Evaluator(cfg)
  ev.files = [standing] * n
  env, wm, base, _, _, _ = ev._build(
    ev.files, drop_terminations=("anchor_pos", "anchor_ori")
  )
  st_path = Path(a.run_dir) / "snapshots" / a.snapshot if a.snapshot else None
  if st_path:
    st = torch.load(st_path, map_location=ev.device)
  else:
    st = {
      "world_model": torch.load(
        Path(a.run_dir) / "world_model.pt", map_location=ev.device
      ),
      "actor": torch.load(Path(a.run_dir) / "actor.pt", map_location=ev.device),
    }
  wm.load_bundle(st["world_model"])
  raw = json.loads((Path(a.run_dir) / "run_config.json").read_text())
  hidden = tuple(
    raw["runner_cfg"].get("velocity", {}).get("encoder_hidden", (256, 256))
  )
  actor = VelocityActor(base, hidden, align_weight=0.0).to(ev.device)
  actor.load_state_dict(st["actor"])
  actor.eval()
  rep, uw = wm.representation, env.unwrapped
  robot = uw.scene["robot"]
  cmd = cmds.to(ev.device).repeat_interleave(TRIALS, 0)
  hist_len = ev.runner_cfg.history_length
  env.reset()
  history = None
  steps, tail = int(HOLD_S / DT), int(TAIL_S / DT)
  err_tail = torch.zeros(n, 3, device=ev.device)
  fell = torch.zeros(n, dtype=torch.bool, device=ev.device)
  for t in range(steps):
    obs = env.get_observations()
    full = obs["full_state"].to(ev.device)
    rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
    sensor = rep.compute_actor_sensor(
      rep_state,
      wm.sim_context.read(uw),
      aux_state=aux,
      sim_actor_obs=obs["actor"].to(ev.device),
      noisy=True,
    )
    history = (
      sensor.unsqueeze(1).repeat(1, hist_len, 1)
      if history is None
      else torch.cat([history[:, 1:], sensor.unsqueeze(1)], 1)
    )
    td = TensorDict({"history": history, "cmd": cmd}, batch_size=[n], device=ev.device)
    env.step(actor(td))
    d = robot.data
    pos = d.root_link_pos_w - uw.scene.env_origins
    fell |= fall_error(pos, d.root_link_quat_w) >= 1.0
    if t >= steps - tail:
      meas = measured_cmd_space(
        d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w
      )
      err_tail += (meas - cmd) / tail
  rel = (err_tail.norm(dim=-1) / cmd.norm(dim=-1)).view(n_cmd, TRIALS)
  fell = fell.view(n_cmd, TRIALS)
  per_cmd, all_fell = [], []
  for i in range(n_cmd):
    ok = rel[i][~fell[i]]
    per_cmd.append(float(ok.median()) if ok.numel() else float("nan"))
    all_fell.append(bool(fell[i].all()))
  vals = np.array(per_cmd)
  summary = {
    "responses": n_cmd,
    "median_rel_error": float(np.nanmedian(vals)),
    "commands_all_trials_fell": int(sum(all_fell)),
    "trials_fell": int(fell.sum()),
  }
  print(json.dumps(summary))
  rows = [
    {"cmd": cmds[i].tolist(), "rel_error": per_cmd[i], "all_fell": all_fell[i]}
    for i in range(n_cmd)
  ]
  out = Path(a.output or Path(a.run_dir) / "velocity_step_test.json")
  out.write_text(json.dumps({"summary": summary, "per_command": rows}, indent=1))
  print(f"[INFO] wrote {out}")
  env.close()


def cli() -> None:
  """Console entry point ``sm-velocity-step-test``."""
  main(tyro.cli(Args))


if __name__ == "__main__":
  cli()
