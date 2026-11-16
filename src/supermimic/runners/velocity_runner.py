"""Velocity-interface runner: trains a velocity encoder (RL or distillation) on a
frozen motion tracker, through the continually refit world model.

See ``supermimic.velocity`` for the paper mapping. Sim envs all use the
velocity interface; the reference motion only supplies reset states and the
reference-sourced commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.policy_learner import ActorCriticPolicyLearner
from supermimic.models import build_mlp
from supermimic.replay_buffer import FieldSpec
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  build_actor,
)
from supermimic.velocity import (
  VelocityActor,
  VelocityCfg,
  cmd_scale,
  fall_error,
  measured_cmd_space,
  reference_cmd,
  sample_uniform_cmd,
  sigmoid_done,
  velocity_reward,
)


class VelocityWMEnv(WorldModelEnv):
  """WM env whose actor goal is a velocity command."""

  def __init__(self, *args, vcfg: VelocityCfg, torso_index: int, **kw):
    self.vcfg = vcfg
    self._torso = torso_index
    super().__init__(*args, **kw)
    if vcfg.mode == "rl":
      self.max_step_reward = self.representation.dt * (2.0 + 2.0)

  def _on_reset(self, batch: TensorDict) -> None:
    self.cmd = batch["vel_cmd"]
    self.cmd_ref = batch["vel_cmd_ref"]
    super()._on_reset(batch)

  def _build_task_goals(self) -> None:
    ref = reference_cmd(self.motion, self.motion_times)
    self.cmd = torch.where(self.cmd_ref[:, None], ref, self.cmd)

  def get_observations(self) -> TensorDict:
    scale = cmd_scale(self.cmd.device)
    critic = torch.cat(
      [self.rep_state, self._wm_goal_obs, self.cmd / scale, self.sim_ctx], -1
    )
    obs = {
      "history": self._sensor_history(),
      "goal_motion": self._actor_goal_obs,
      "cmd": self.cmd,
      "cmd_ref": self.cmd_ref,
      "critic": critic,
    }
    return TensorDict(obs, batch_size=[self.cmd.shape[0]], device=self.cmd.device)

  def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
    if self.vcfg.mode != "rl":
      return super()._wm_step(rep_state, aux_state, action, last_action, sim_ctx)
    next_aux, next_rep = self.dynamics(rep_state, action, aux_state, sim_ctx)
    next_joint = self.representation.joint_state(next_rep)
    _, rot, lin, ang = self.representation.world_frame_bodies(next_aux, next_rep)
    root_quat = next_aux[..., 3:7]
    reward = velocity_reward(
      root_quat,
      lin[:, 0],
      ang[:, 0],
      rot[:, self._torso],
      ang[:, self._torso],
      next_joint.chunk(2, dim=-1)[0],
      action,
      last_action,
      self.cmd,
      self.representation._soft_joint_pos_limits,
      self.representation.dt,
    )
    term = sigmoid_done(
      fall_error(next_aux[..., :3], root_quat), self._termination_softness
    )
    return next_aux, next_rep, next_joint, term, reward


@dataclass
class VelocityRunnerCfg(SuperMimicRunnerCfg):
  velocity: VelocityCfg = field(default_factory=VelocityCfg)


def load_base_cfg(run_dir: str) -> SuperMimicRunnerCfg:
  raw = json.loads((Path(run_dir) / "run_config.json").read_text())
  return load_runner_cfg(raw)


class VelocityRunner(SuperMimicRunner):
  cfg: VelocityRunnerCfg  # type: ignore[assignment]

  def _init_policy(self) -> None:
    v = self.cfg.velocity
    snap = torch.load(
      Path(v.init_run_dir) / "snapshots" / v.init_snapshot, map_location=self.device
    )
    self._wm.load_bundle(snap["world_model"])
    base = build_actor(self.cfg, self._wm, self.device)
    base.load_state_dict(snap["actor"])
    actor = VelocityActor(base, v.encoder_hidden, v.align_weight).to(self.device)

    rep = self._representation
    critic_in = rep.rep_state_dim + self._num_goals + 3 + self._sim_context.output_dim
    critic = build_mlp(critic_in, self.cfg.critic_hidden_dims, output_dim=1).to(
      self.device
    )

    torso = list(self.motion_cmd.cfg.body_names).index("torso_link")
    self.wm_env = VelocityWMEnv(
      dynamics=self._dynamics,
      representation=rep,
      env=self.env,
      batch_size=self.cfg.policy.batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
      vcfg=v,
      torso_index=torso,
    )
    if torch.cuda.is_available():
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)
    self._sampler = None
    self.policy_learner = ActorCriticPolicyLearner(
      actor, critic, self.wm_env, self.cfg.policy, device=self.device
    )
    n = self.env.num_envs
    self._cmd = torch.zeros(n, 3, device=self.device)
    self._cmd_ref = torch.ones(n, dtype=torch.bool, device=self.device)
    self._cmd_age = torch.zeros(n, dtype=torch.long, device=self.device)
    self._resample(torch.ones(n, dtype=torch.bool, device=self.device))
    self._err_sum = torch.zeros(3, device=self.device)
    self._err_n = 0

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    schema = super()._buffer_schema()
    schema["vel_cmd"] = FieldSpec(3)
    schema["vel_cmd_ref"] = FieldSpec(dtype=torch.bool)
    return schema

  def _resample(self, mask: torch.Tensor) -> None:
    v = self.cfg.velocity
    n = self.env.num_envs
    p_ref = 1.0 if v.mode == "distill" else v.p_ref
    ref = torch.rand(n, device=self.device) < p_ref
    self._cmd_ref = torch.where(mask, ref, self._cmd_ref)
    self._cmd = torch.where(
      mask[:, None], sample_uniform_cmd(n, v.p_zero, self.device), self._cmd
    )
    self._cmd_age = torch.where(mask, torch.zeros_like(self._cmd_age), self._cmd_age)

  def _update_actor_normalization(self, actor_inputs) -> None:
    del actor_inputs

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)
      times = self.motion_cmd.time_steps.to(self.device).clone()
      rep_state = self._representation.represent(curr_states)
      aux = self._representation.extract_aux_state(curr_states)
      sim_ctx = self._sim_context.read(self.env.unwrapped)

      self._resample(self._cmd_age >= self.cfg.velocity.resample_steps)
      ref = reference_cmd(self.motion, times)
      self._cmd = torch.where(self._cmd_ref[:, None], ref, self._cmd)
      sensor = self._representation.compute_actor_sensor(
        rep_state, sim_ctx, aux_state=aux, sim_actor_obs=sim_actor_obs, noisy=True
      )
      _, goal_motion = self._representation.compute_wm_goal(self.motion, times, aux)
      td = TensorDict(
        {
          "history": self._collection_history(sensor),
          "goal_motion": goal_motion,
          "cmd": self._cmd.clone(),
          "cmd_ref": self._cmd_ref.clone(),
        },
        batch_size=[self.env.num_envs],
        device=self.device,
      )
      actions = self.policy_learner.actor(td)
      actions = actions + torch.randn_like(actions) * self._collection_noise_std()
      extra = {
        "sim_ctx": sim_ctx,
        "vel_cmd": self._cmd.clone(),
        "vel_cmd_ref": self._cmd_ref.clone(),
      }
      if self._stores_actor_obs:
        extra["actor_obs"] = sim_actor_obs
      next_obs, rewards, dones, extras = self._step_and_store(
        actions, curr_states, actor_td=td, extra_fields=extra
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

      d = self.env.unwrapped.scene["robot"].data
      meas = measured_cmd_space(
        d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w
      )
      self._err_sum += (meas - self._cmd).abs().mean(0)
      self._err_n += 1
      self._cmd_age += 1
      self._resample(dones.to(self.device).bool())
      if self._is_main and self._iteration % 1000 == 0 and self._err_n:
        e = (self._err_sum / self._err_n).tolist()
        print(
          f"[vel] it {self._iteration}: sim |err| vx {e[0]:.3f} vy {e[1]:.3f} wz {e[2]:.3f} "
          f"(ref share {self._cmd_ref.float().mean():.2f})",
          flush=True,
        )
        self._err_sum.zero_()
        self._err_n = 0
    self._flush_normalization()
    return self.env.num_envs
