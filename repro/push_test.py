"""Push-recovery test (paper Sec. 4.3): hold a static T-pose, get a horizontal
root-velocity push, and see whether the policy settles or falls.

Setup (our concrete choices; the paper gives only "0.5-4 m/s velocity pushes
while holding a static T-pose in simulation"):
  reference  LAFAN1 walk1_subject1 frame 0 (a T-pose) held static for 12 s
  push       at t = 3 s, Delta v added to the root linear velocity, horizontal,
             8 directions x magnitudes {0.5, 1, 2, 3, 4} m/s, one env each
  sim        friction 1.0, no DR, no obs noise, deterministic actor, tracking
             terminations disabled (we want to watch stumbling)
  fell       root height < 0.35 m or 1 + u_z > 0.7 (paper Table A.6 fall rule)
  settled    root speed < 0.1 m/s and joint RMS error to the T-pose within
             0.03 rad of the env's own pre-push (2-3 s) level, continuously
             until the end; settle time = push -> last entry

Usage:
  uv run repro/push_test.py --run-dir <run> [--snapshot it0040000.pt]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.tasks.task_spec import build_actor_obs_td

_spec = importlib.util.spec_from_file_location(
  "repro_evaluate", Path(__file__).with_name("evaluate.py")
)
assert _spec is not None and _spec.loader is not None
ev_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev_mod
_spec.loader.exec_module(ev_mod)

DT = 0.02
SPEEDS = (0.5, 1.0, 2.0, 3.0, 4.0)
N_DIRS = 8
PUSH_T, END_T = 3.0, 12.0


def make_tpose(src: str, out: str, seconds: float = END_T + 1.0) -> str:
  d = dict(np.load(src))
  n = int(seconds / DT)
  tp = {
    "fps": d["fps"],
    "joint_names": d["joint_names"],
    "joint_pos": np.repeat(d["joint_pos"][:1], n, 0),
    "joint_vel": np.zeros((n, d["joint_vel"].shape[1]), np.float32),
    "root_pos_w": np.repeat(d["root_pos_w"][:1], n, 0),
    "root_quat_w": np.repeat(d["root_quat_w"][:1], n, 0),
    "root_lin_vel_w": np.zeros((n, 3), np.float32),
    "root_ang_vel_w": np.zeros((n, 3), np.float32),
  }
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **tp)
  return out


@dataclass
class Args:
  run_dir: str
  snapshot: str | None = None
  tpose_src: str = "../../data/lafan1_g1/npz/walk1_subject1.npz"
  tpose_out: str = "../../data/tpose/tpose_static.npz"
  output: str | None = None
  device: str | None = None


@torch.inference_mode()
def main(a: Args) -> None:
  tpose = make_tpose(a.tpose_src, a.tpose_out)
  n_env = N_DIRS * len(SPEEDS)
  cfg = ev_mod.EvalConfig(run_dir=a.run_dir, motion_file=tpose, device=a.device)
  ev = ev_mod.Evaluator(cfg)
  ev.files = [tpose] * n_env
  env, wm, actor, cmd, _, _ = ev._build(ev.files, drop_terminations=("anchor_pos", "anchor_ori"))
  if a.snapshot:
    st = torch.load(Path(a.run_dir) / "snapshots" / a.snapshot, map_location=ev.device)
  else:
    st = {
      "world_model": torch.load(Path(a.run_dir) / "world_model.pt", map_location=ev.device),
      "actor": torch.load(Path(a.run_dir) / "actor.pt", map_location=ev.device),
    }
  wm.load_bundle(st["world_model"])
  actor.load_state_dict(st["actor"])
  actor.eval()
  actor.set_slot_mix({"motion": 1.0})
  rep, uw = wm.representation, env.unwrapped
  robot = uw.scene["robot"]
  specs = [ts for ts in actor.task_specs if ts.name == "motion"]
  hist_len = ev.runner_cfg.history_length
  proprio = ev.runner_cfg.actor_obs == "proprio"

  ang = torch.arange(N_DIRS, device=ev.device) * (2 * torch.pi / N_DIRS)
  mag = torch.tensor(SPEEDS, device=ev.device).repeat_interleave(N_DIRS)
  ang = ang.repeat(len(SPEEDS))
  push = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], -1)

  q_ref = torch.as_tensor(np.load(tpose)["joint_pos"][0], device=ev.device)
  env.reset()
  history = None
  steps = int(END_T / DT)
  push_step = int(PUSH_T / DT)
  speed_log, jerr_log, fell = [], [], torch.zeros(n_env, dtype=torch.bool, device=ev.device)
  for t in range(steps):
    obs = env.get_observations()
    full = obs["full_state"].to(ev.device)
    rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
    sensor = rep.compute_actor_sensor(
      rep_state,
      wm.sim_context.read(uw),
      aux_state=aux,
      sim_actor_obs=obs["actor"].to(ev.device) if proprio else None,
      noisy=False,
    )
    history = (
      sensor.unsqueeze(1).repeat(1, hist_len, 1)
      if history is None
      else torch.cat([history[:, 1:], sensor.unsqueeze(1)], 1)
    )
    goals = {ts.name: ts.extract_training_goal(cmd.motion, cmd.time_steps.clone(), aux) for ts in specs}
    env.step(actor(build_actor_obs_td(goals, history)))
    if t == push_step:
      vel = robot.data.root_link_vel_w.clone()
      vel[:, :2] += push
      robot.write_root_link_velocity_to_sim(vel)
    d = robot.data
    speed_log.append(d.root_link_lin_vel_w[:, :2].norm(dim=-1))
    jerr_log.append((d.joint_pos - q_ref).pow(2).mean(-1).sqrt())
    g_b = d.projected_gravity_b  # unit gravity in root frame; upright -> (0,0,-1)
    fell |= (d.root_link_pos_w[:, 2] < 0.35) | (1 + g_b[:, 2] > 0.7)
  speed_all, jerr_all = torch.stack(speed_log, 1), torch.stack(jerr_log, 1)
  pre = slice(push_step - 50, push_step)
  print(
    f"[diag] pre-push (2-3 s): speed med {speed_all[:, pre].median():.3f} "
    f"max {speed_all[:, pre].max():.3f} | joint RMS err med {jerr_all[:, pre].median():.3f} "
    f"max {jerr_all[:, pre].max():.3f}"
  )
  print(
    f"[diag] last 1 s: speed med {speed_all[:, -50:].median():.3f} | "
    f"joint err med {jerr_all[:, -50:].median():.3f}"
  )
  speed = speed_all[:, push_step:]
  jerr = jerr_all[:, push_step:]
  # Relative to each env's own pre-push hold: the policy holds the T-pose with a
  # steady joint offset (~0.1 rad RMS for LAFAN1 policies), so an absolute
  # threshold would call every env unsettled.
  base = jerr_all[:, pre].mean(1, keepdim=True)
  ok = (speed < 0.1) & (jerr < base + 0.03)
  # last index where not ok, +1 -> settle step (relative to push)
  not_ok = (~ok).float()
  idx = torch.arange(not_ok.shape[1], device=ev.device).float()
  last_bad = torch.where(not_ok > 0, idx, torch.full_like(idx, -1)).max(1).values
  settled = (last_bad < not_ok.shape[1] - 1) & ~fell
  settle_s = torch.where(settled, (last_bad + 1) * DT, torch.full_like(last_bad, float("nan")))
  rows = [
    {
      "speed": float(mag[i]),
      "dir_deg": float(torch.rad2deg(ang[i])),
      "fell": bool(fell[i]),
      "settled": bool(settled[i]),
      "settle_s": None if not settled[i] else round(float(settle_s[i]), 2),
    }
    for i in range(n_env)
  ]
  print(f"{'speed':>6} {'fell':>5} {'settled':>8} {'settle s (min/median/max)':>28}")
  for s in SPEEDS:
    r = [x for x in rows if x["speed"] == s]
    ts = [x["settle_s"] for x in r if x["settle_s"] is not None]
    rng = f"{min(ts):.2f}/{np.median(ts):.2f}/{max(ts):.2f}" if ts else "-"
    print(
      f"{s:6.1f} {sum(x['fell'] for x in r):3d}/{len(r)} "
      f"{sum(x['settled'] for x in r):4d}/{len(r)} {rng:>28}"
    )
  out = Path(a.output or Path(a.run_dir) / "push_test.json")
  out.write_text(json.dumps(rows, indent=1))
  print(f"[INFO] wrote {out}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
