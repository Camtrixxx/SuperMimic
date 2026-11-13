"""Shared learner utilities."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict


def build_optimizers(
  modules: Sequence[nn.Module],
  lr: float,
  betas: tuple[float, float] = (0.9, 0.999),
  device: str = "cpu",
) -> list[torch.optim.Optimizer]:
  """Muon on the Linear weights between the first and the last Linear of each
  ``nn.Sequential`` in ``modules``, fused capturable Adam on the rest, both at
  ``lr``. Muon's ``lr`` is a float, fixed once an update is captured."""
  hidden = [
    linear.weight
    for module in modules
    for seq in module.modules()
    if isinstance(seq, nn.Sequential)
    for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
  ]
  hidden_ids = {id(p) for p in hidden}
  rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
  adam = torch.optim.Adam(
    rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True
  )
  if not hidden:
    return [adam]
  return [
    adam,
    ScheduledMuon(hidden, lr=lr, device=device, adjust_lr_fn="match_rms_adamw"),
  ]


class ScheduledMuon(torch.optim.Muon):
  """``torch.optim.Muon`` whose learning rate lives in a device tensor
  (``group["lr_t"]``), so a CUDA-graph-captured step follows in-place changes.

  repro: torch's Muon converts ``lr`` to a Python scalar on every step, which
  freezes it at capture time and makes LR schedules a silent no-op. Same math as
  ``torch.optim._muon._single_tensor_muon``.
  """

  def __init__(self, params, lr: float, device: str | torch.device = "cpu", **kw):
    super().__init__(params, lr=lr, **kw)
    for group in self.param_groups:
      group["lr_t"] = torch.tensor(float(lr), device=device)

  @torch.no_grad()
  def step(self, closure=None):  # noqa: D102
    from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

    assert closure is None
    for group in self.param_groups:
      lr_t, wd, mom = group["lr_t"], group["weight_decay"], group["momentum"]
      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p]
        if "momentum_buffer" not in state:
          state["momentum_buffer"] = torch.zeros_like(p.grad)
        buf = state["momentum_buffer"]
        buf.lerp_(p.grad, 1 - mom)
        update = p.grad.lerp(buf, mom) if group["nesterov"] else buf
        update = _zeropower_via_newtonschulz(
          update, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        ratio = _adjust_lr(1.0, group["adjust_lr_fn"], p.shape)
        p.mul_(1 - lr_t * wd)
        p.sub_(update * (lr_t * ratio))


def set_lr_scale(optimizers: Iterable[torch.optim.Optimizer], scale: float) -> None:
  """Scale every optimizer's LR to ``base_lr * scale`` in place (graph-safe)."""
  for opt in optimizers:
    for group in opt.param_groups:
      base = group.setdefault("base_lr", float(group["lr"]))
      key = "lr_t" if "lr_t" in group else "lr"
      if torch.is_tensor(group[key]):
        group[key].fill_(base * scale)
      else:
        raise RuntimeError(f"{type(opt).__name__} lr is not a tensor; cannot schedule")


def lr_scale(step: int, schedule: str, decay_steps: int, final_ratio: float) -> float:
  """Paper Table A.5: cosine decay to ``final_ratio`` over ``decay_steps``, then
  constant. ``schedule='constant'`` keeps the official repo's behaviour."""
  if schedule == "constant":
    return 1.0
  assert schedule == "cosine", schedule
  t = min(step, decay_steps) / decay_steps
  return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def _allreduce_grads(params: list[torch.Tensor]) -> None:
  """repro: data parallel (paper Table A.5: "gradients averaged across GPUs").
  One flat all-reduce, before clipping, so every rank takes the same step.
  No-op without an initialized multi-rank process group. Capture-safe (NCCL
  collectives can be recorded in CUDA graphs)."""
  if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
    return
  grads = [p.grad for p in params if p.grad is not None]
  if not grads:
    return
  flat = torch.cat([g.reshape(-1) for g in grads])
  dist.all_reduce(flat)
  flat.div_(dist.get_world_size())
  offset = 0
  for g in grads:
    n = g.numel()
    g.copy_(flat[offset : offset + n].view_as(g))
    offset += n


def broadcast_module(*modules: nn.Module) -> None:
  """Copy rank 0's parameters and buffers to every rank."""
  if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
    return
  for m in modules:
    for t in [*m.parameters(), *m.buffers()]:
      dist.broadcast(t.data, src=0)


def _optimize_step(
  optimizers: Sequence[torch.optim.Optimizer],
  loss: torch.Tensor,
  params: Iterable[torch.Tensor] | None = None,
  max_norm: float | None = None,
  clip_value: float = 0.0,
) -> torch.Tensor | None:
  """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
  for optimizer in optimizers:
    optimizer.zero_grad(set_to_none=True)
  loss.backward()
  if params is not None:
    params = list(params)
    _allreduce_grads(params)
  grad_norm = None
  if params is not None and max_norm is not None:
    # Don't set error_if_nonfinite=True: it forces a per-step CPU<->GPU sync.
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if clip_value > 0:
      torch.nn.utils.clip_grad_value_(params, clip_value)
  for optimizer in optimizers:
    optimizer.step()
  return grad_norm


def graph_state(
  modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]
) -> list[torch.Tensor]:
  """What a captured update reads that Python can rebind: parameters, buffers,
  optimizer state and learning-rate tensors."""
  state: list[torch.Tensor] = []
  for module in modules:
    state += [*module.parameters(), *module.buffers()]
  for optimizer in optimizers:
    state += [t for s in optimizer.state.values() for t in s.values()]
    state += [
      group[k]
      for group in optimizer.param_groups
      for k in ("lr", "lr_t")
      if k in group and torch.is_tensor(group[k])
    ]
  return state


def _identity(state: list[torch.Tensor]) -> list[int]:
  return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
  return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
  """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as
  a CUDA graph and replays it on the device of ``state()``. ``fn`` must be
  capture-safe: no host syncs, fixed shapes, and every tensor it reads besides
  its inputs updated in place. Python values it reads are fixed at capture, and
  each replay overwrites the outputs it returns. All CapturedUpdates on a device
  record on one stream into one memory pool and must replay in the order they
  were recorded. A replay raises if a tensor in ``state()`` moved; a learning rate
  that changes is a device tensor, changed in place with ``group["lr"].fill_(lr)``."""

  def __init__(
    self,
    fn: Callable[[TensorDict], dict[str, torch.Tensor]],
    state: Callable[[], list[torch.Tensor]],
    warmup: int = 3,
  ):
    self._fn = fn
    self._state = state
    self._warmup = warmup
    self._inputs: TensorDict | None = None
    self._graph: torch.cuda.CUDAGraph | None = None
    self._identity: list[int] = []
    self._outputs: dict[str, torch.Tensor] = {}

  def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
    state = self._state()
    if not state[0].is_cuda:
      return self._fn(inputs)
    with torch.cuda.device(state[0].device):
      stream, pool = _capture_stream_and_pool(state[0].device)
      if self._inputs is None:
        self._inputs = inputs.clone()
      else:
        self._inputs.update_(inputs)
      if self._warmup > 0:
        self._warmup -= 1
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
          outputs = self._fn(self._inputs)
        torch.cuda.current_stream().wait_stream(stream)
        return outputs
      if self._graph is None:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
          self._outputs = self._fn(self._inputs)
        self._graph, self._identity = graph, _identity(self._state())
      elif _identity(state) != self._identity:
        raise RuntimeError("a captured tensor moved")
      self._graph.replay()
      torch.cuda.current_stream().synchronize()
      return self._outputs
