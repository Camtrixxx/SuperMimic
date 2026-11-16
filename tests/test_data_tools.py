"""Data tools on synthetic inputs (no dataset or GPU needed)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from mjlab.scripts.csv_to_npz import JOINT_NAMES

from supermimic import paths
from supermimic.data import bones_filter, convert, feasibility


def _bones_csv(path, n=13, joints_shuffled=True):
  cols = {
    "Frame": np.arange(n),
    "root_translateX": np.full(n, 100.0),  # cm
    "root_translateY": np.full(n, 200.0),
    "root_translateZ": np.full(n, 79.0),
    "root_rotateX": np.zeros(n),  # deg
    "root_rotateY": np.zeros(n),
    "root_rotateZ": np.full(n, 90.0),
  }
  names = list(JOINT_NAMES)
  if joints_shuffled:
    names = names[::-1]  # name matching, not column order
  for j in names:
    cols[f"{j}_dof"] = np.full(n, math.degrees(0.01 * (JOINT_NAMES.index(j) + 1)))
  pd.DataFrame(cols).to_csv(path, index=False)


def test_bones_csv_units_and_joint_order(tmp_path):
  p = tmp_path / "clip.csv"
  _bones_csv(p)
  out = convert.convert(str(p), 120.0, 50.0, "bones")
  np.testing.assert_allclose(
    out["root_pos_w"], [[1.0, 2.0, 0.79]] * len(out["root_pos_w"]), atol=1e-6
  )
  h = math.sqrt(0.5)  # 90 deg about z, wxyz
  np.testing.assert_allclose(np.abs(out["root_quat_w"][0]), [h, 0, 0, h], atol=1e-6)
  expect = 0.01 * (np.arange(len(JOINT_NAMES)) + 1)
  np.testing.assert_allclose(out["joint_pos"][0], expect, atol=1e-6)
  np.testing.assert_allclose(out["joint_vel"], 0.0, atol=1e-5)
  # 12 input intervals at 120 fps = 0.1 s -> 6 frames at 50 Hz (endpoint included)
  assert out["joint_pos"].shape == (6, len(JOINT_NAMES))


def test_name_filter(tmp_path):
  names = [
    "walk_forward_001__A001",
    "sit_on_chair_001__A001",
    "jump_off_50cm_001__A002",
    "look_over_fence_180_R_001__A003",
    "Box_HSPU_001__A004",  # SONIC keyword, mixed case
    "neutral_button press_001__A005",  # names may contain spaces
  ]
  m = pd.DataFrame(
    {"filename": names, "move_g1_path": [f"g1/csv/240101/{n}.csv" for n in names]}
  )
  m.to_parquet(tmp_path / "meta.parquet")
  keep, removed = tmp_path / "keep.txt", tmp_path / "removed.txt"
  bones_filter.main(
    bones_filter.Args(str(tmp_path / "meta.parquet"), str(keep), str(removed))
  )
  assert keep.read_text().splitlines() == [
    "walk_forward_001__A001",
    "neutral_button press_001__A005",
  ]
  assert len(removed.read_text().splitlines()) == 4


def _signals(n, **over):
  sig = {
    "zmin": np.full(n, 0.03),
    "foot_l": np.full(n, -0.004),
    "foot_r": np.full(n, -0.004),
    "pelvis_h": np.full(n, 0.79),
    "up_z": np.full(n, 1.0),
    "speed": np.zeros(n),
  }
  sig.update(over)
  return sig


def test_feasibility_rules():
  n = 200
  assert not feasibility.judge(_signals(n))["infeasible"]
  # elevated support: z_min > 0.20 m for >= 0.6 s (30 frames), not 29
  z = np.full(n, 0.03)
  z[10:40] = 0.25
  assert feasibility.judge(_signals(n, zmin=z))["elevated"]
  z[39] = 0.03
  assert not feasibility.judge(_signals(n, zmin=z))["elevated"]
  # ground penetration: any frame < -0.12, or > 80% of frames < -0.08
  z = np.full(n, 0.03)
  z[5] = -0.13
  assert feasibility.judge(_signals(n, zmin=z))["penetration"]
  z = np.full(n, -0.09)
  assert feasibility.judge(_signals(n, zmin=z))["penetration"]
  # floating feet: both feet > 2 cm for >= 0.5 s while standing still upright
  f = np.full(n, -0.004)
  f[50:80] = 0.05
  sig = _signals(n, foot_l=f.copy(), foot_r=f.copy())
  assert feasibility.judge(sig)["floating"]
  sig["speed"][50:80] = 1.0  # a ballistic jump is not "floating"
  assert not feasibility.judge(sig)["floating"]


def test_data_root_env_override(monkeypatch, tmp_path):
  monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
  assert paths.data_path("bones_seed") == tmp_path / "bones_seed"
  monkeypatch.delenv(paths.ENV_VAR)
  assert paths.data_root() == paths.REPO_ROOT / "data"
  with pytest.raises(SystemExit):
    paths.require(tmp_path / "missing", "thing")
