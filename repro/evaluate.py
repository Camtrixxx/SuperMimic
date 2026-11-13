"""Paper Appendix F.1 benchmark for SuperMimic checkpoints, without W&B.

Protocol: each clip plays once from its first frame; foot friction 1.0, no
domain randomization, no observation noise, deterministic actor. A clip fails
if it triggers the training termination (anchor height / tilt). Normalized
return = undiscounted reward sum / (steps in clip * max per-step reward).

One env per clip, so all clips run in parallel; ``--max-envs`` chunks large
libraries.

Example:
  uv run repro/evaluate.py --run-dir logs/SuperMimic-GenCoord-G1/<ts> \
    --motion-file 'data/lafan1_g1/npz/*.npz'
  # every snapshot (return-vs-time curve):
  uv run repro/evaluate.py --run-dir ... --motion-file ... --all-snapshots
"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.tasks.task_spec import build_actor_obs_td


@dataclass
class EvalConfig:
  run_dir: str
  """Training log dir holding run_config.json (+ actor.pt / snapshots/)."""
  motion_file: str
  """Glob or comma list of clip .npz files to benchmark."""
  task_id: str = "SuperMimic-GenCoord-G1"
  snapshot: str | None = None
  """Snapshot file name in run_dir/snapshots; default: latest actor.pt."""
  all_snapshots: bool = False
  """Evaluate every snapshot and write a return-vs-wall-time curve."""
  every: int = 1
  """With --all-snapshots, evaluate every n-th snapshot."""
  max_envs: int = 4096
  device: str | None = None
  output: str | None = None
  shard: int = 0
  """Evaluate chunks shard, shard+num_shards, ... (one process per GPU)."""
  num_shards: int = 1
  sort_by_length: bool = False
  """Group similar-length clips into chunks (use for large libraries)."""


def _clip_len(path: str) -> int:
  with np.load(path) as z:
    return int(z["joint_pos"].shape[0])


def _resolve_files(spec: str) -> list[str]:
  out: list[str] = []
  for part in spec.split(","):
    part = os.path.expanduser(part.strip())
    out += sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part]
  if not out:
    raise SystemExit(f"no motion files match {spec}")
  return out


def _eval_env_cfg(task_id: str, files: list[str]):
  cfg = load_env_cfg(task_id, play=True)  # infinite episode, no pushes/corruption
  cfg.commands["motion"].motion_file = files
  cfg.commands["motion"].sampling_mode = "external"  # overridden per-env below
  cfg.commands["motion"].pose_range = {}
  cfg.commands["motion"].velocity_range = {}
  cfg.commands["motion"].joint_position_range = (0.0, 0.0)
  ev = cfg.events
  # Keep the DR events (SimContext discovers its specs from them) but pin them
  # to nominal values. The ctx normalization comes from the checkpoint.
  ev["foot_friction"].params["ranges"] = (1.0, 1.0)
  ev["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
  ev["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.observations["actor"].enable_corruption = False
  return cfg


class Evaluator:
  def __init__(self, cfg: EvalConfig):
    configure_torch_backends()
    self.cfg = cfg
    self.device = cfg.device or "cuda:0"
    run_dir = Path(cfg.run_dir)
    raw = json.loads((run_dir / "run_config.json").read_text())
    self.runner_cfg = load_runner_cfg(raw)
    check_representation_matches(cfg.task_id, self.runner_cfg.representation)
    self.files = _resolve_files(cfg.motion_file)
    self.run_dir = run_dir

  def _build(self, files: list[str]):
    env_cfg = _eval_env_cfg(self.cfg.task_id, files)
    env_cfg.scene.num_envs = len(files)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=self.device))
    wm = build_world_model(self.runner_cfg, env, self.device)
    actor = build_actor(self.runner_cfg, wm, self.device)
    cmd = env.unwrapped.command_manager.get_term("motion")
    ends = cmd.motion.motion_ends
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    # Env i always resets to the first frame of clip i.
    cmd._external_sampling = lambda env_ids: starts[env_ids]
    return env, wm, actor, cmd, starts, ends

  @torch.inference_mode()
  def run_chunk(self, files: list[str], states: list[dict]) -> list[list[dict]]:
    env, wm, actor, cmd, starts, ends = self._build(files)
    rep = wm.representation
    uw = env.unwrapped
    n = len(files)
    max_step_reward = uw.step_dt * sum(
      t.weight for t in uw.cfg.rewards.values() if t.weight > 0.0
    )
    clip_steps = (ends - starts - 1).float()  # steps until motion_end fires
    active_specs = [
      ts
      for ts in actor.task_specs
      if self.runner_cfg.slot_mix.get(ts.name, 0.0) > 0.0 and ts.name == "motion"
    ]
    actor.set_slot_mix({"motion": 1.0})
    hist_len = self.runner_cfg.history_length
    proprio = self.runner_cfg.actor_obs == "proprio"
    results: list[list[dict]] = []
    for state in states:
      wm.load_bundle(state["world_model"])
      actor.load_state_dict(state["actor"])
      actor.eval()
      env.reset()
      ret = torch.zeros(n, device=self.device)
      alive = torch.ones(n, dtype=torch.bool, device=self.device)
      failed = torch.zeros(n, dtype=torch.bool, device=self.device)
      history = None
      for _ in range(int(clip_steps.max().item()) + 1):
        obs = env.get_observations()
        full = obs["full_state"].to(self.device)
        rep_state = rep.represent(full)
        aux = rep.extract_aux_state(full)
        sim_ctx = wm.sim_context.read(uw)
        sensor = rep.compute_actor_sensor(
          rep_state,
          sim_ctx,
          aux_state=aux,
          sim_actor_obs=obs["actor"].to(self.device) if proprio else None,
          noisy=False,
        )
        if history is None:
          history = sensor.unsqueeze(1).repeat(1, hist_len, 1)
        else:
          history = torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
        times = cmd.time_steps.clone()
        goals = {
          ts.name: ts.extract_training_goal(cmd.motion, times, aux)
          for ts in active_specs
        }
        actions = actor(build_actor_obs_td(goals, history))
        _, rew, dones, extras = env.step(actions)
        ret += torch.where(alive, rew.to(self.device), 0.0)
        tm = uw.termination_manager
        term = tm.terminated.to(self.device).bool() & ~tm.time_outs.bool()
        failed |= alive & term
        alive &= ~dones.to(self.device).bool()
        if not alive.any():
          break
      norm = (ret / (clip_steps * max_step_reward)).clamp(max=1.0)
      results.append(
        [
          {
            "clip": Path(f).stem,
            "frames": int(clip_steps[i].item()) + 1,
            "normalized_return": float(norm[i]),
            "failed": bool(failed[i]),
          }
          for i, f in enumerate(files)
        ]
      )
    env.close()
    return results


def _summary(rows: list[dict]) -> dict:
  n = len(rows)
  frames = sum(r["frames"] for r in rows)
  return {
    "clips": n,
    "failed": sum(r["failed"] for r in rows),
    "normalized_return_mean": sum(r["normalized_return"] for r in rows) / n,
    "normalized_return_frame_weighted": sum(
      r["normalized_return"] * r["frames"] for r in rows
    )
    / frames,
  }


def main(cfg: EvalConfig) -> None:
  ev = Evaluator(cfg)
  snap_dir = ev.run_dir / "snapshots"
  if cfg.all_snapshots:
    paths = sorted(snap_dir.glob("it*.pt"))[:: cfg.every]
  elif cfg.snapshot:
    paths = [snap_dir / cfg.snapshot]
  else:
    paths = []
  if paths:
    states = [torch.load(p, map_location=ev.device) for p in paths]
  else:
    states = [
      {
        "iteration": None,
        "wall_time_s": None,
        "world_model": torch.load(ev.run_dir / "world_model.pt", map_location=ev.device),
        "actor": torch.load(ev.run_dir / "actor.pt", map_location=ev.device),
      }
    ]
  per_state: list[list[dict]] = [[] for _ in states]
  files = ev.files
  if cfg.sort_by_length:
    # Similar-length clips per chunk, so a chunk is not held open by one long clip.
    with ThreadPoolExecutor(32) as pool:
      lengths = list(pool.map(_clip_len, files))
    files = [f for _, f in sorted(zip(lengths, files, strict=True))]
  chunks = [files[lo : lo + cfg.max_envs] for lo in range(0, len(files), cfg.max_envs)]
  chunks = chunks[cfg.shard :: cfg.num_shards]
  for k, chunk in enumerate(chunks):
    for i, rows in enumerate(ev.run_chunk(chunk, states)):
      per_state[i] += rows
    print(f"[INFO] shard {cfg.shard}: chunk {k + 1}/{len(chunks)} done", flush=True)
  report = []
  for st, rows in zip(states, per_state, strict=True):
    entry = {
      "iteration": st.get("iteration"),
      "wall_time_h": None if st.get("wall_time_s") is None else st["wall_time_s"] / 3600,
      **_summary(rows),
      "failed_clips": [r["clip"] for r in rows if r["failed"]],
    }
    report.append({**entry, "per_clip": rows})
    print(
      f"it={entry['iteration']} t={entry['wall_time_h']} "
      f"ret={entry['normalized_return_mean']:.4f} "
      f"(frame-w {entry['normalized_return_frame_weighted']:.4f}) "
      f"failed={entry['failed']}/{entry['clips']}"
    )
  out = Path(cfg.output or ev.run_dir / "eval.json")
  out.write_text(json.dumps(report, indent=1))
  print(f"[INFO] wrote {out}")


if __name__ == "__main__":
  main(tyro.cli(EvalConfig))
