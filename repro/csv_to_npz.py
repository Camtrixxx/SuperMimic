"""Batch-convert retargeted G1 CSV motions to mjlab tracking ``.npz`` files.

Vectorized, local-output equivalent of ``mjlab/scripts/csv_to_npz.py`` (which
converts one file per call with a per-frame Python slerp and uploads to W&B).
Arrays and resampling match it exactly; see ``repro/test_csv_to_npz.py``.

CSV rows: ``[root_pos(3), root_quat_xyzw(4), joint_pos(29)]``.

Example:
  uv run repro/csv_to_npz.py --input-glob 'data/lafan1_g1/csv/*.csv' \
    --output-dir data/lafan1_g1/npz --input-fps 30
"""

from __future__ import annotations

import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from mjlab.scripts.csv_to_npz import JOINT_NAMES
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)


def batched_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
  """Row-wise slerp with the same branches as ``mjlab...math.quat_slerp``."""
  eps = torch.finfo(q1.dtype).eps * 4.0
  q1_in, q2_in = q1, q2
  d = (q1 * q2).sum(-1)
  # Same expression order as the reference: the |d| == 1 test sits right at
  # eps on real data, so a different rounding path flips the branch.
  near_one = (d.abs() - 1.0).abs() < eps
  q2 = torch.where((d < 0.0).unsqueeze(-1), -q2, q2)
  d = d.abs()
  angle = torch.acos(d.clamp(-1, 1))
  safe = angle.clamp(min=eps)
  isin = 1.0 / torch.sin(safe)
  t = tau.unsqueeze(-1)
  out = q1 * torch.sin((1.0 - t) * safe.unsqueeze(-1)) * isin.unsqueeze(-1)
  out = out + q2 * torch.sin(t * safe.unsqueeze(-1)) * isin.unsqueeze(-1)
  keep_q1 = (tau == 0.0) | near_one | (angle.abs() < eps)
  out = torch.where(keep_q1.unsqueeze(-1), q1, out)
  # The reference computes ``d`` with torch.dot, whose float32 rounding no
  # batched op reproduces, and branches on ``|d| - 1 < eps``. Rows within a few
  # ULP of that edge are recomputed with the reference itself (rare: only
  # near-identical consecutive frames reach it).
  edge = ((d - 1.0).abs() - eps).abs() < 8 * eps
  for i in torch.nonzero(edge & (tau != 0.0)).flatten().tolist():
    out[i] = quat_slerp(q1_in[i].clone(), q2_in[i].clone(), float(tau[i]))
  # tau == 1 returns the original (un-flipped) q2 in the reference.
  return out


def convert(path: str, input_fps: float, output_fps: float) -> dict[str, np.ndarray]:
  motion = torch.from_numpy(np.loadtxt(path, delimiter=",", ndmin=2)).float()
  pos_in = motion[:, :3]
  rot_in = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
  dof_in = motion[:, 7 : 7 + len(JOINT_NAMES)]
  n_in = motion.shape[0]
  in_dt, out_dt = 1.0 / input_fps, 1.0 / output_fps
  duration = (n_in - 1) * in_dt

  num_frames = math.floor(duration / out_dt + 1e-6) + 1
  times = torch.arange(num_frames, dtype=torch.float32) * out_dt
  phase = times / duration
  i0 = (phase * (n_in - 1)).floor().long()
  i1 = torch.minimum(i0 + 1, torch.tensor(n_in - 1))
  blend = phase * (n_in - 1) - i0

  pos = pos_in[i0] * (1 - blend[:, None]) + pos_in[i1] * blend[:, None]
  dof = dof_in[i0] * (1 - blend[:, None]) + dof_in[i1] * blend[:, None]
  rot = batched_slerp(rot_in[i0], rot_in[i1], blend)
  rot = torch.where((blend == 1.0)[:, None], rot_in[i1], rot)

  lin_vel = torch.gradient(pos, spacing=out_dt, dim=0)[0]
  dof_vel = torch.gradient(dof, spacing=out_dt, dim=0)[0]
  q_rel = quat_mul(rot[2:], quat_conjugate(rot[:-2]))
  omega = axis_angle_from_quat(q_rel) / (2.0 * out_dt)
  ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  return {
    "fps": np.array([output_fps]),
    "joint_names": np.array(JOINT_NAMES),
    "joint_pos": dof.numpy(),
    "joint_vel": dof_vel.numpy(),
    "root_pos_w": pos.numpy(),
    "root_quat_w": rot.numpy(),
    "root_lin_vel_w": lin_vel.numpy(),
    "root_ang_vel_w": ang_vel.numpy(),
  }


@dataclass
class Args:
  input_glob: str
  output_dir: str
  input_fps: float = 30.0
  output_fps: float = 50.0
  workers: int = 32
  overwrite: bool = False


def _job(job: tuple[str, str, float, float]) -> str:
  src, dst, input_fps, output_fps = job
  torch.set_num_threads(1)
  np.savez(dst, **convert(src, input_fps, output_fps))
  return dst


def main(args: Args) -> None:
  files = sorted(glob.glob(os.path.expanduser(args.input_glob), recursive=True))
  if not files:
    raise SystemExit(f"no files match {args.input_glob}")
  os.makedirs(args.output_dir, exist_ok=True)
  jobs = []
  for src in files:
    dst = os.path.join(args.output_dir, os.path.splitext(os.path.basename(src))[0])
    if args.overwrite or not os.path.exists(dst + ".npz"):
      jobs.append((src, dst + ".npz", args.input_fps, args.output_fps))
  print(f"[INFO] {len(files)} files, {len(jobs)} to convert -> {args.output_dir}")
  with ProcessPoolExecutor(args.workers) as pool:
    for i, _ in enumerate(pool.map(_job, jobs, chunksize=8), 1):
      if i % 1000 == 0 or i == len(jobs):
        print(f"[INFO] {i}/{len(jobs)}")


if __name__ == "__main__":
  main(tyro.cli(Args))
