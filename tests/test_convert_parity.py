"""Parity: supermimic.data.convert == mjlab.scripts.csv_to_npz.MotionLoader.

Needs real CSVs: SM_PARITY_CSV='data/lafan1_g1/csv/*.csv' uv run pytest tests/test_convert_parity.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
from mjlab.scripts.csv_to_npz import MotionLoader

from supermimic.data import convert as repro

FILES = sorted(glob.glob(os.environ.get("SM_PARITY_CSV", "")))

# Finite-difference velocities divide float32 roundoff (~1e-7 on positions,
# ~4e-7 on quaternions after a different-but-equivalent op order) by 2*dt=0.04.
ATOL = {"root_ang_vel_w": 5e-5, "root_lin_vel_w": 2e-5, "joint_vel": 2e-5}


@pytest.mark.skipif(not FILES, reason="set SM_PARITY_CSV to a CSV glob")
@pytest.mark.parametrize("path", FILES)
def test_matches_reference(path: str) -> None:
  ref = MotionLoader(path, input_fps=30.0, output_fps=50.0).arrays()
  out = repro.convert(path, 30.0, 50.0)
  assert ref.keys() == out.keys()
  assert (ref["joint_names"] == out["joint_names"]).all()
  for key in ref:
    if key == "joint_names":
      continue
    assert ref[key].shape == out[key].shape, key
    atol = ATOL.get(key, 1e-6)
    np.testing.assert_allclose(out[key], ref[key], atol=atol, rtol=0, err_msg=key)
