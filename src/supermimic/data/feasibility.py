# pyright: reportAttributeAccessIssue=false
# (mujoco is a compiled extension without type stubs)
"""Paper Appendix E feasibility filter for G1 motion clips (50 Hz npz).

Per frame, MuJoCo forward kinematics on the G1 model plus a z=0 floor gives
  z_min(t)   lowest body-link origin height,
  foot_l/r   exact foot-collision-geom-to-floor distance (mj_geomDistance,
             min over each foot's 7 capsules).
A clip is infeasible if any rule fires:
  elevated support   z_min > 0.20 m for >= 0.6 s
  ground penetration z_min < -0.12 m anywhere, or < -0.08 m on > 80% of frames
  floating feet      both feet > 0.02 m off the floor for >= 0.5 s while the
                     pelvis is upright, at standing height and nearly still
The paper does not quantify "upright / standing height / nearly stationary";
the gates below are our choice (see STANDING_* constants).

Usage:
  uv run sm-feasibility --npz-glob 'data/.../npz/*.npz' --out feas.parquet
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import mujoco
import numpy as np
import pandas as pd
import tyro
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

FPS = 50.0
# Paper thresholds (Appendix E).
ELEVATED_Z, ELEVATED_S = 0.20, 0.6
PENETRATE_Z, PENETRATE_SOFT_Z, PENETRATE_SOFT_FRAC = -0.12, -0.08, 0.8
FLOAT_Z, FLOAT_S = 0.02, 0.5
# Our gates for "pelvis upright, at standing height, body nearly stationary".
STANDING_UP_Z = 0.9  # pelvis z-axis vertical component (cos ~26 deg)
STANDING_MIN_H = 0.65  # pelvis height [m]; nominal G1 standing ~0.79
STANDING_MAX_SPEED = 0.25  # root linear speed [m/s]

_MODEL: mujoco.MjModel | None = None
_FOOT_GEOMS: tuple[list[int], list[int]] | None = None
_FLOOR = -1
_QADR: np.ndarray | None = None


def _build() -> None:
  global _MODEL, _FOOT_GEOMS, _FLOOR, _QADR
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  floor = spec.worldbody.add_geom()
  floor.name = "repro_floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = [0, 0, 0.05]
  m = spec.compile()
  _MODEL = m
  _FLOOR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "repro_floor")
  feet = []
  for side in ("left", "right"):
    feet.append(
      [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{i}_collision")
        for i in range(1, 8)
      ]
    )
  assert all(g >= 0 for f in feet for g in f), "foot geoms not found"
  _FOOT_GEOMS = (feet[0], feet[1])
  from mjlab.scripts.csv_to_npz import JOINT_NAMES

  _QADR = np.array(
    [
      m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
      for j in JOINT_NAMES
    ]
  )


def frame_signals(clip: dict) -> dict[str, np.ndarray]:
  """Per-frame z_min, foot distances, pelvis height / up-z / speed."""
  if _MODEL is None:
    _build()
  assert _MODEL is not None and _FOOT_GEOMS is not None and _QADR is not None
  m, (lf, rf), qadr = _MODEL, _FOOT_GEOMS, _QADR
  d = mujoco.MjData(m)
  pos, quat, q = clip["root_pos_w"], clip["root_quat_w"], clip["joint_pos"]
  n = q.shape[0]
  zmin = np.empty(n)
  foot = np.empty((n, 2))
  up_z = np.empty(n)
  fromto = np.empty(6)
  for t in range(n):
    d.qpos[:3] = pos[t]
    d.qpos[3:7] = quat[t]
    d.qpos[qadr] = q[t]
    mujoco.mj_kinematics(m, d)
    zmin[t] = d.xpos[1:, 2].min()  # body 0 is the world
    for k, geoms in enumerate((lf, rf)):
      foot[t, k] = min(
        mujoco.mj_geomDistance(m, d, g, _FLOOR, 1.0, fromto) for g in geoms
      )
    up_z[t] = d.xmat[1].reshape(3, 3)[2, 2]  # pelvis z-axis, world z component
  speed = np.linalg.norm(clip["root_lin_vel_w"], axis=-1)
  return {
    "zmin": zmin,
    "foot_l": foot[:, 0],
    "foot_r": foot[:, 1],
    "pelvis_h": pos[:, 2],
    "up_z": up_z,
    "speed": speed,
  }


def _longest_run(mask: np.ndarray) -> int:
  best = cur = 0
  for v in mask:
    cur = cur + 1 if v else 0
    best = max(best, cur)
  return best


def judge(sig: dict[str, np.ndarray]) -> dict:
  zmin = sig["zmin"]
  elevated = _longest_run(zmin > ELEVATED_Z) >= round(ELEVATED_S * FPS)
  penetration = bool(
    (zmin < PENETRATE_Z).any() or (zmin < PENETRATE_SOFT_Z).mean() > PENETRATE_SOFT_FRAC
  )
  standing = (
    (sig["up_z"] > STANDING_UP_Z)
    & (sig["pelvis_h"] > STANDING_MIN_H)
    & (sig["speed"] < STANDING_MAX_SPEED)
  )
  both_up = (sig["foot_l"] > FLOAT_Z) & (sig["foot_r"] > FLOAT_Z)
  floating = _longest_run(both_up & standing) >= round(FLOAT_S * FPS)
  lowest_foot = np.minimum(sig["foot_l"], sig["foot_r"])
  return {
    "elevated": bool(elevated),
    "penetration": penetration,
    "floating": bool(floating),
    "infeasible": bool(elevated or penetration or floating),
    "zmin_min": float(zmin.min()),
    "zmin_max_run_s": _longest_run(zmin > ELEVATED_Z) / FPS,
    "float_run_s": _longest_run(both_up & standing) / FPS,
    # stance calibration: lowest foot while standing (paper: -0.004 +- 0.004 m)
    "stance_foot_med": float(np.median(lowest_foot[standing]))
    if standing.any()
    else np.nan,
    "frames": int(zmin.shape[0]),
  }


def _job(path: str) -> dict:
  clip = dict(np.load(path))
  return {
    "clip": os.path.splitext(os.path.basename(path))[0],
    **judge(frame_signals(clip)),
  }


@dataclass
class Args:
  npz_glob: str
  out: str
  workers: int = 64
  limit: int | None = None


def main(a: Args) -> None:
  files = sorted(glob.glob(os.path.expanduser(a.npz_glob)))[: a.limit]
  rows = []
  with ProcessPoolExecutor(a.workers) as pool:
    for i, r in enumerate(pool.map(_job, files, chunksize=16), 1):
      rows.append(r)
      if i % 5000 == 0 or i == len(files):
        print(f"[INFO] {i}/{len(files)}", flush=True)
  df = pd.DataFrame(rows)
  df.to_parquet(a.out)
  print(df[["elevated", "penetration", "floating", "infeasible"]].sum().to_string())


def cli() -> None:
  """Console entry point ``sm-feasibility``."""
  main(tyro.cli(Args))


if __name__ == "__main__":
  cli()
