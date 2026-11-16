"""Evaluation helpers and the velocity-interface math (no dataset or GPU)."""

from __future__ import annotations

import json
import math
import types

import pytest
import torch

from supermimic.evaluation import merge
from supermimic.velocity import fall_error, reference_cmd, velocity_reward


def _shard(path, clips):
  rows = [
    {"clip": c, "frames": 100, "normalized_return": r, "failed": f} for c, r, f in clips
  ]
  path.write_text(
    json.dumps([{"iteration": 8000, "wall_time_h": 0.5, "per_clip": rows}])
  )


def test_merge_shards_and_groups(tmp_path, capsys):
  _shard(
    tmp_path / "s0.json", [("flip_180_R_001", 0.2, True), ("walk_001", 0.9, False)]
  )
  _shard(
    tmp_path / "s1.json", [("side_roll_002", 0.8, False), ("walk_002", 1.0, False)]
  )
  out = tmp_path / "merged.json"
  merge.main(merge.Args(pattern=str(tmp_path / "s*.json"), out=str(out)))
  rep = json.loads(out.read_text())[0]
  assert rep["clips"] == 4 and rep["failed"] == 1
  assert rep["failed_clips"] == ["flip_180_R_001"]
  assert rep["normalized_return_mean"] == pytest.approx(0.725)
  assert rep["groups"]["flips (360/180/90)"] == [1, 1]
  assert rep["groups"]["side rolls"] == [0, 1]
  assert "failed=1" in capsys.readouterr().out


def _identity(n):
  return torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)


def test_velocity_reward_terms():
  n, dt = 3, 0.02
  q = _identity(n)
  cmd = torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1)
  zeros3, a = torch.zeros(n, 3), torch.zeros(n, 29)
  rot = torch.eye(3).expand(n, 3, 3).clone()
  lim = torch.stack([-torch.ones(29), torch.ones(29)], -1)
  r = velocity_reward(q, cmd.clone(), zeros3, rot, zeros3, a, a, a, cmd, lim, dt)
  torch.testing.assert_close(r, torch.full((n,), dt * 4.0))  # 2 + 2, no penalties
  lean = rot.clone()
  lean[:, :, 2] = torch.tensor([0.0, math.sqrt(0.5), math.sqrt(0.5)])  # 45 deg sideways
  r2 = velocity_reward(q, cmd.clone(), zeros3, lean, zeros3, a, a, a, cmd, lim, dt)
  torch.testing.assert_close(r2, torch.full((n,), dt * (4.0 - 4.0 * 0.5)))


def test_reference_cmd_heading_frame():
  t = 40
  yaw90 = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
  motion = types.SimpleNamespace(
    motion_ends=torch.tensor([t]),
    root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(t, 1),
    root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.5]]).repeat(t, 1),
    root_quat_w=yaw90.repeat(t, 1),
  )
  cmd = reference_cmd(motion, torch.tensor([0, 20, t - 1]))
  # world +x seen from a robot facing +y is its -y
  torch.testing.assert_close(
    cmd, torch.tensor([[0.0, -1.0, 0.5]]).repeat(3, 1), atol=1e-6, rtol=0
  )


def test_fall_error():
  up = _identity(2)
  pos = torch.tensor([[0.0, 0.0, 0.79], [0.0, 0.0, 0.30]])
  err = fall_error(pos, up)
  assert err[0] < 1.0 <= err[1]
