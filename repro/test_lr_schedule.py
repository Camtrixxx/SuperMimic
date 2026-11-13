"""ScheduledMuon parity with torch Muon, and LR changes reaching a captured graph.

Run: uv run pytest repro/test_lr_schedule.py
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.learner_utils import (
  CapturedUpdate,
  ScheduledMuon,
  _optimize_step,
  build_optimizers,
  graph_state,
  lr_scale,
  set_lr_scale,
)


def test_matches_torch_muon() -> None:
  torch.manual_seed(0)
  a = nn.Parameter(torch.randn(64, 32))
  b = nn.Parameter(a.detach().clone())
  ref = torch.optim.Muon([a], lr=2e-4, adjust_lr_fn="match_rms_adamw")
  ours = ScheduledMuon([b], lr=2e-4, adjust_lr_fn="match_rms_adamw")
  for _ in range(5):
    g = torch.randn(64, 32)
    a.grad, b.grad = g.clone(), g.clone()
    ref.step()
    ours.step()
  # torch scales by an fp64 Python float, we by an fp32 tensor: ~1e-7 roundoff
  # per step, against steps of ~1e-4.
  torch.testing.assert_close(b, a, rtol=0, atol=2e-6)


def test_cosine_endpoints() -> None:
  assert lr_scale(0, "cosine", 250_000, 0.03) == pytest.approx(1.0)
  assert lr_scale(125_000, "cosine", 250_000, 0.03) == pytest.approx(0.515)
  assert lr_scale(250_000, "cosine", 250_000, 0.03) == pytest.approx(0.03)
  assert lr_scale(320_000, "cosine", 250_000, 0.03) == pytest.approx(0.03)
  assert lr_scale(10, "constant", 250_000, 0.03) == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lr_change_reaches_captured_graph() -> None:
  torch.manual_seed(0)
  net = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 16), nn.Linear(16, 1)).cuda()
  opts = build_optimizers([net], 1e-2, device="cuda")
  assert any(isinstance(o, ScheduledMuon) for o in opts)

  def fn(td: TensorDict) -> dict[str, torch.Tensor]:
    loss = net(td["x"]).pow(2).mean()
    _optimize_step(opts, loss, net.parameters(), max_norm=1.0)
    return {"loss": loss.detach()}

  upd = CapturedUpdate(fn, lambda: graph_state([net], opts), warmup=1)
  x = TensorDict({"x": torch.randn(32, 4, device="cuda")}, batch_size=[32])
  upd(x)  # warmup
  upd(x)  # capture + replay
  hidden = net[1].weight

  def delta(scale: float) -> float:
    set_lr_scale(opts, scale)
    before = hidden.detach().clone()
    upd(x)
    return (hidden.detach() - before).norm().item()

  d1, d0 = delta(1.0), delta(0.0)
  assert d1 > 0 and math.isclose(d0, 0.0, abs_tol=1e-9), (d1, d0)
