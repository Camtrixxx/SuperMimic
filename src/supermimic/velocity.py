"""Velocity-command interface (paper Sec. 4.5, Appendix B and G).

The official release only ships the distillation variant (velocity goal read
from the reference, motion-tracking reward). This module adds the paper's RL
variant and a matched distillation baseline, trained as a downstream encoder on
a frozen, already-trained motion tracker. The paper notes (Sec. 6) that because
the velocity reward never updates the shared decoder, concurrent training "is
the same as training the velocity encoder after the motion tracker".

Pieces (paper reference in brackets):
  command   (vx, vy) planar root velocity in the heading frame + world yaw rate
            [G.1]; reference commands are the centered 25-frame mean of the
            reference root velocity in the reference's heading frame
  sampling  at reset and every 250 steps: from the reference w.p. 0.5, else
            uniform vx in [-1.5, 2], vy in [-1.32, 1.32], wz in [-3.35, 3.35],
            zeroed w.p. 0.1 [G.1]
  encoder   MLP(256, 256) on [normalized command, last-10-frame history] -> z
            [Table A.4]; frozen shared decoder (stop-gradient on its weights)
  reward    Table A.6 velocity tracking + regularizers, all x dt   (RL mode)
  done      fall only: root height < 0.35 m or 1 + u_z > 0.7      (RL mode)
  align     0.1 * ||z_vel - sg(z_motion)||^2 on reference-command rollouts
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from tensordict import TensorDict

from supermimic.models import MultiEncoderAE, build_mlp

CMD_LO = (-1.5, -1.32, -3.35)
CMD_HI = (2.0, 1.32, 3.35)
CMD_SCALE = (2.0, 1.32, 3.35)  # per-axis normalization: max |range|
REF_WINDOW = 25
TORSO = "torso_link"


@dataclass
class VelocityCfg:
  mode: str = "rl"
  """'rl' = velocity-tracking reward + fall-only termination (paper's RL encoder);
  'distill' = motion-tracking reward, commands always from the reference."""
  p_ref: float = 0.5
  p_zero: float = 0.1
  resample_steps: int = 250
  align_weight: float = 0.1
  encoder_hidden: tuple[int, ...] = (256, 256)
  init_run_dir: str = ""
  """Trained motion-tracker run (run_config.json + snapshots/) to start from."""
  init_snapshot: str = ""
  """Snapshot file name in init_run_dir/snapshots, e.g. it0112000.pt."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _bounds(device) -> tuple[torch.Tensor, torch.Tensor]:
  return torch.tensor(CMD_LO, device=device), torch.tensor(CMD_HI, device=device)


def cmd_scale(device) -> torch.Tensor:
  return torch.tensor(CMD_SCALE, device=device)


def sample_uniform_cmd(n: int, p_zero: float, device) -> torch.Tensor:
  lo, hi = _bounds(device)
  cmd = lo + torch.rand(n, 3, device=device) * (hi - lo)
  zero = torch.rand(n, device=device) < p_zero
  return torch.where(zero[:, None], torch.zeros_like(cmd), cmd)


def heading_frame_planar(quat: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
  """World vector -> (x, y) in the yaw-only frame of ``quat``."""
  return quat_apply_inverse(yaw_quat(quat), vec_w)[..., :2]


def reference_cmd(motion, times: torch.Tensor) -> torch.Tensor:
  """Centered 25-frame mean of the reference root twist, clamped inside each clip;
  linear part in the reference heading frame at ``times``, yaw rate in world."""
  ends = motion.motion_ends
  clip = torch.searchsorted(ends, times, right=True).clamp(max=ends.numel() - 1)
  start = torch.where(clip > 0, ends[(clip - 1).clamp(min=0)], torch.zeros_like(clip))
  last = ends[clip] - 1
  half = REF_WINDOW // 2
  offs = torch.arange(REF_WINDOW, device=times.device) - half
  idx = torch.minimum(
    torch.maximum(times[:, None] + offs, start[:, None]), last[:, None]
  )
  v = motion.root_lin_vel_w[idx].mean(1)
  wz = motion.root_ang_vel_w[idx][..., 2].mean(1)
  vxy = heading_frame_planar(motion.root_quat_w[times], v)
  lo, hi = _bounds(times.device)
  return torch.clamp(torch.cat([vxy, wz[:, None]], -1), lo, hi)


def measured_cmd_space(
  root_quat: torch.Tensor, lin_vel_w: torch.Tensor, ang_vel_w: torch.Tensor
) -> torch.Tensor:
  """Robot (vx, vy) in its heading frame and world yaw rate."""
  return torch.cat(
    [heading_frame_planar(root_quat, lin_vel_w), ang_vel_w[..., 2:3]], -1
  )


# ---------------------------------------------------------------------------
# Reward and termination (Table A.6), on world-frame quantities
# ---------------------------------------------------------------------------


def velocity_reward(
  root_quat: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
  torso_rot_w: torch.Tensor,
  torso_ang_vel_w: torch.Tensor,
  joint_pos: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  cmd: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  meas = measured_cmd_space(root_quat, root_lin_vel_w, root_ang_vel_w)
  lin = torch.exp(-((meas[..., :2] - cmd[..., :2]) ** 2).sum(-1) / 0.5**2)
  yaw = torch.exp(-((meas[..., 2] - cmd[..., 2]) ** 2) / 0.5)
  smooth = ((action - last_action) ** 2).sum(-1)
  lo, hi = soft_joint_pos_limits.unbind(-1)
  limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  hq = yaw_quat(root_quat)
  torso_z_h = quat_apply_inverse(
    hq, torso_rot_w[..., :, 2]
  )  # torso z-axis, heading frame
  lean = torso_z_h[..., 1] ** 2  # sin^2 of the sideways lean (forward lean free)
  w_h = quat_apply_inverse(hq, torso_ang_vel_w)
  rp_rate = w_h[..., 0] ** 2 + w_h[..., 1] ** 2
  return dt * (
    2.0 * lin + 2.0 * yaw - 0.3 * smooth - 2.0 * limit - 4.0 * lean - 0.05 * rp_rate
  )


def fall_error(root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
  """Normalized fall measure; >= 1 means fallen (h < 0.35 m or 1 + u_z > 0.7)."""
  g_w = torch.zeros_like(root_pos)
  g_w[..., 2] = -1.0
  u_z = quat_apply_inverse(root_quat, g_w)[..., 2]
  height = 0.35 / root_pos[..., 2].clamp(min=1e-3)
  return torch.maximum(height, (1.0 + u_z) / 0.7)


# ---------------------------------------------------------------------------
# Actor: velocity encoder + frozen motion tracker
# ---------------------------------------------------------------------------


class VelocityActor(nn.Module):
  """``a = dec(tanh(enc_vel([cmd/scale, hist_n])), hist_n)`` with the motion
  tracker (decoder, normalizers, motion encoder) frozen."""

  def __init__(
    self, base: MultiEncoderAE, hidden: tuple[int, ...], align_weight: float
  ):
    super().__init__()
    self.base = base
    base.requires_grad_(False)
    base.eval()
    hist_dim = base.history_length * base.obs_dim
    self.encoder = build_mlp(3 + hist_dim, hidden, output_dim=base.latent_dim)
    self.register_buffer("cmd_scale", torch.tensor(CMD_SCALE))
    self.align_weight = align_weight
    self._aux = torch.zeros(0)

  def train(self, mode: bool = True):  # keep the frozen tracker in eval mode
    super().train(mode)
    self.base.eval()
    return self

  def forward(self, td: TensorDict) -> torch.Tensor:
    hist = self.base.obs_normalizer(td["history"]).flatten(start_dim=1)
    z = torch.tanh(self.encoder(torch.cat([td["cmd"] / self.cmd_scale, hist], -1)))
    if self.training and self.align_weight > 0:
      with torch.no_grad():
        gm = self.base.goal_normalizers["motion"](td["goal_motion"])
        z_m = torch.tanh(self.base.encoders["motion"](gm))
      mask = td["cmd_ref"].float()
      self._aux = self.align_weight * ((z - z_m) ** 2).mean(-1) * mask
    else:
      self._aux = torch.zeros(z.shape[0], device=z.device)
    return self.base.decoder(torch.cat([z, hist], -1))

  def auxiliary_loss(self) -> torch.Tensor:
    return self._aux

  def auxiliary_stats(self) -> dict[str, torch.Tensor]:
    return {"align_loss": self._aux.mean().detach()} if self._aux.numel() else {}

  @torch.jit.unused
  def update_normalization(self, td: TensorDict) -> None:
    del td  # frozen tracker normalizers


def sigmoid_done(err: torch.Tensor, softness: float) -> torch.Tensor:
  return torch.sigmoid((err - 1.0) / softness)


def fall_termination(env) -> torch.Tensor:
  """mjlab termination term: fall rule of Table A.6 on the simulated robot."""
  d = env.scene["robot"].data
  return (
    fall_error(d.root_link_pos_w - env.scene.env_origins, d.root_link_quat_w) >= 1.0
  )
