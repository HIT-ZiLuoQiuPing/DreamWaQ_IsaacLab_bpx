"""Small PPO runner for DreamWaQ-style IsaacLab training."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
import torch.optim as optim

from .actor_critic import DreamWaQActorCritic
from .config import DreamWaQConfig
from .storage import DreamWaQRolloutStorage


class DreamWaQRunner:
    """PPO runner that trains a CENet context estimator together with the policy."""

    def __init__(self, env, cfg: DreamWaQConfig, log_dir: str | None = None, device: str = "cpu"):
        self.env = env
        self.cfg = cfg
        self.alg_cfg = cfg.algorithm
        self.policy_cfg = cfg.policy
        self.log_dir = log_dir
        self.device = device
        self.current_learning_iteration = 0
        self.tot_time = 0.0
        self.tot_timesteps = 0
        self.writer = None
        self._last_rollout_stats: dict[str, float | None] = {}
        self._last_reward_metrics: dict[str, float] = {}

        obs, extras = self._get_initial_observations()
        groups = self._get_observation_groups(extras)
        history = self._require_group(groups, "cenet")
        critic_obs = groups.get("critic", obs)
        estimator_target = self._require_group(groups, "estimator")

        if estimator_target.shape[1] < 3:
            raise ValueError(
                "DreamWaQ estimator observation group must start with 3D base linear velocity."
            )

        self.policy = DreamWaQActorCritic(
            obs.shape[1],
            history.shape[1],
            critic_obs.shape[1],
            estimator_target.shape[1],
            self.env.num_actions,
            **self.policy_cfg.__dict__,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.alg_cfg.learning_rate)
        self.storage = DreamWaQRolloutStorage(
            self.env.num_envs,
            self.cfg.num_steps_per_env,
            obs.shape[1],
            history.shape[1],
            critic_obs.shape[1],
            estimator_target.shape[1],
            self.env.num_actions,
            self.device,
        )
        print(
            "[INFO] DreamWaQ observation dims: "
            f"policy={obs.shape[1]}, cenet={history.shape[1]}, "
            f"critic={critic_obs.shape[1]}, estimator={estimator_target.shape[1]}, "
            f"actions={self.env.num_actions}"
        )

    def _get_initial_observations(self) -> tuple[torch.Tensor, dict]:
        result = self.env.get_observations()
        if isinstance(result, tuple):
            obs, extras = result
        else:
            obs, extras = result, {}
        return obs.to(self.device), extras

    @staticmethod
    def _get_observation_groups(extras: dict) -> dict[str, torch.Tensor]:
        observations = extras.get("observations", {}) if isinstance(extras, dict) else {}
        return observations if observations is not None else {}

    @staticmethod
    def _require_group(groups: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name not in groups:
            available = ", ".join(sorted(groups.keys())) or "none"
            raise KeyError(f"Missing DreamWaQ observation group '{name}'. Available groups: {available}.")
        return groups[name]

    def _unpack_step(
        self,
        obs: torch.Tensor,
        infos: dict,
        fallback_critic_obs: torch.Tensor,
        fallback_estimator_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = self._get_observation_groups(infos)
        next_obs = obs.to(self.device)
        next_history = self._require_group(groups, "cenet").to(self.device)
        next_critic_obs = groups.get("critic", fallback_critic_obs).to(self.device)
        next_estimator_target = groups.get("estimator", fallback_estimator_target).to(self.device)
        return next_obs, next_history, next_critic_obs, next_estimator_target

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf"):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self._get_initial_observations()
        groups = self._get_observation_groups(extras)
        history = self._require_group(groups, "cenet").to(self.device)
        critic_obs = groups.get("critic", obs).to(self.device)
        estimator_target = self._require_group(groups, "estimator").to(self.device)

        self.policy.train()
        rewbuffer: deque[float] = deque(maxlen=100)
        lenbuffer: deque[float] = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        total_iterations = self.current_learning_iteration + num_learning_iterations
        learning_start_iteration = self.current_learning_iteration
        for iteration in range(self.current_learning_iteration, total_iterations):
            start = time.time()
            ep_infos = []
            done_count = 0
            timeout_count = 0
            iter_completed_rewards: list[float] = []
            iter_completed_lengths: list[float] = []
            iter_reward_terms: dict[str, float] = {}
            iter_reward_term_count = 0
            action_abs_sum = 0.0
            action_mean_abs_sum = 0.0
            action_noise_abs_sum = 0.0
            command_x_sum = 0.0
            velocity_x_sum = 0.0
            rollout_stat_count = 0

            with torch.inference_mode():
                for _ in range(self.cfg.num_steps_per_env):
                    self.policy.update_normalization(obs, history, critic_obs)
                    episode_lengths_before_step = self._episode_lengths_for_done()
                    actions = self.policy.act(obs, history).detach()
                    values = self.policy.evaluate(critic_obs).detach()
                    actions_log_prob = self.policy.get_actions_log_prob(actions).detach()
                    action_mean = self.policy.action_mean.detach()
                    action_sigma = self.policy.action_std.detach()
                    action_abs_sum += float(actions.abs().mean().item())
                    action_mean_abs_sum += float(action_mean.abs().mean().item())
                    action_noise_abs_sum += float((actions - action_mean).abs().mean().item())
                    command = self._current_command()
                    if command is not None:
                        command_x_sum += float(command[:, 0].mean().item())
                    velocity_x_sum += float(estimator_target[:, 0].mean().item())
                    rollout_stat_count += 1

                    next_obs_raw, rewards, dones, infos = self.env.step(actions)
                    for name, value in self._reward_step_values().items():
                        iter_reward_terms[name] = iter_reward_terms.get(name, 0.0) + value
                    iter_reward_term_count += 1
                    next_obs, next_history, next_critic_obs, next_estimator_target = self._unpack_step(
                        next_obs_raw, infos, critic_obs, estimator_target
                    )

                    rewards = rewards.to(self.device).view(-1)
                    dones = dones.to(self.device).view(-1).bool()
                    done_count += int(dones.sum().item())
                    if "time_outs" in infos:
                        time_outs = infos["time_outs"].to(self.device).view(-1, 1)
                        timeout_count += int(time_outs.sum().item())
                        rewards += self.alg_cfg.gamma * torch.squeeze(values * time_outs, dim=1)

                    self.storage.add(
                        observations=obs,
                        histories=history,
                        critic_observations=critic_obs,
                        estimator_targets=estimator_target,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        values=values,
                        actions_log_prob=actions_log_prob,
                        action_mean=action_mean,
                        action_sigma=action_sigma,
                    )

                    log_info = {}
                    if isinstance(infos.get("episode"), dict):
                        log_info.update(infos["episode"])
                    if isinstance(infos.get("log"), dict):
                        log_info.update(infos["log"])
                    if log_info:
                        ep_infos.append(log_info)
                    cur_reward_sum += rewards
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_ids) > 0:
                        completed_rewards = cur_reward_sum[done_ids].cpu().numpy().tolist()
                        completed_lengths = episode_lengths_before_step[done_ids].cpu().numpy().tolist()
                        rewbuffer.extend(completed_rewards)
                        lenbuffer.extend(completed_lengths)
                        iter_completed_rewards.extend(completed_rewards)
                        iter_completed_lengths.extend(completed_lengths)
                        cur_reward_sum[done_ids] = 0

                    obs = next_obs
                    history = next_history
                    critic_obs = next_critic_obs
                    estimator_target = next_estimator_target

                collection_time = time.time() - start
                last_values = self.policy.evaluate(critic_obs).detach()
                self.storage.compute_returns(last_values, self.alg_cfg.gamma, self.alg_cfg.lam)
                rollout_transitions = self.cfg.num_steps_per_env * self.env.num_envs
                self._last_rollout_stats = {
                    "done_rate": done_count / rollout_transitions,
                    "timeout_rate": timeout_count / rollout_transitions,
                    "mean_completed_reward": self._mean_list(iter_completed_rewards),
                    "mean_completed_episode_length": self._mean_list(iter_completed_lengths),
                    "action_abs": action_abs_sum / max(rollout_stat_count, 1),
                    "action_mean_abs": action_mean_abs_sum / max(rollout_stat_count, 1),
                    "action_noise_abs": action_noise_abs_sum / max(rollout_stat_count, 1),
                    "command_x": command_x_sum / max(rollout_stat_count, 1),
                    "velocity_x": velocity_x_sum / max(rollout_stat_count, 1),
                }
                self._last_reward_metrics = {
                    name: value / max(iter_reward_term_count, 1) for name, value in iter_reward_terms.items()
                }

            learn_start = time.time()
            losses = self.update()
            learn_time = time.time() - learn_start
            self.storage.clear()

            self.current_learning_iteration = iteration + 1
            if self.log_dir is not None:
                self._log(
                    iteration,
                    total_iterations,
                    learning_start_iteration,
                    collection_time,
                    learn_time,
                    losses,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                )
            if self.log_dir is not None and iteration % self.cfg.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))

        self.current_learning_iteration = total_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_loss = 0.0
        mean_reconstruction_loss = 0.0
        mean_kl_loss = 0.0
        num_updates = 0

        for batch in self.storage.mini_batch_generator(self.alg_cfg.num_mini_batches, self.alg_cfg.num_learning_epochs):
            (
                obs_batch,
                history_batch,
                critic_obs_batch,
                estimator_target_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch

            self.policy.act(obs_batch, history_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.alg_cfg.desired_kl is not None and self.alg_cfg.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                        / (2.0 * sigma_batch.square())
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.alg_cfg.desired_kl * 2.0:
                        self.alg_cfg.learning_rate = max(1.0e-5, self.alg_cfg.learning_rate / 1.5)
                    elif kl_mean < self.alg_cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.alg_cfg.learning_rate = min(1.0e-2, self.alg_cfg.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.alg_cfg.learning_rate

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.alg_cfg.clip_param, 1.0 + self.alg_cfg.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.alg_cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.alg_cfg.clip_param, self.alg_cfg.clip_param
                )
                value_losses = (value_batch - returns_batch).square()
                value_losses_clipped = (value_clipped - returns_batch).square()
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).square().mean()

            self.policy.update_normalization(obs_batch, history_batch, critic_obs_batch)
            cenet_stats = self.policy.cenet_loss(history_batch, estimator_target_batch)
            loss = (
                surrogate_loss
                + self.alg_cfg.value_loss_coef * value_loss
                - self.alg_cfg.entropy_coef * entropy_batch.mean()
                + self.alg_cfg.estimator_loss_coef * cenet_stats["estimator_velocity"]
                + self.alg_cfg.reconstruction_loss_coef * cenet_stats["estimator_terrain"]
                + self.alg_cfg.kl_loss_coef * cenet_stats["kl"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.alg_cfg.max_grad_norm)
            self.optimizer.step()
            self.policy.clamp_action_std()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += cenet_stats["estimator_velocity"].item()
            mean_reconstruction_loss += cenet_stats["estimator_terrain"].item()
            mean_kl_loss += cenet_stats["kl"].item()
            num_updates += 1

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "estimator": mean_estimator_loss / num_updates,
            "terrain": mean_reconstruction_loss / num_updates,
            "kl": mean_kl_loss / num_updates,
        }

    def _unwrap_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def _current_command(self) -> torch.Tensor | None:
        env = self._unwrap_env()
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return None
        try:
            command = command_manager.get_command("base_velocity")
        except (AttributeError, KeyError):
            return None
        return command.to(self.device) if isinstance(command, torch.Tensor) else None

    def _reward_step_values(self) -> dict[str, float]:
        env = self._unwrap_env()
        reward_manager = getattr(env, "reward_manager", None)
        step_reward = getattr(reward_manager, "_step_reward", None)
        if not isinstance(step_reward, torch.Tensor) or step_reward.ndim != 2:
            return {}

        term_names = getattr(reward_manager, "active_terms", None) or getattr(reward_manager, "_term_names", [])
        if len(term_names) != step_reward.shape[1]:
            return {}

        term_means = step_reward.detach().float().mean(dim=0)
        return {
            f"Reward/step/{term_name}": float(term_means[index].item()) for index, term_name in enumerate(term_names)
        }

    def _episode_lengths_for_done(self) -> torch.Tensor:
        env = self._unwrap_env()
        episode_length_buf = getattr(env, "episode_length_buf", None)
        if episode_length_buf is None:
            return torch.ones(self.env.num_envs, dtype=torch.float, device=self.device)
        return episode_length_buf.to(self.device).float() + 1.0

    @staticmethod
    def _scalar(value) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _mean_episode_infos(self, ep_infos: list[dict]) -> dict[str, float]:
        if not ep_infos:
            return {}
        values: dict[str, list[float]] = {}
        for ep_info in ep_infos:
            for key, value in ep_info.items():
                scalar = self._scalar(value)
                if scalar is not None:
                    values.setdefault(key, []).append(scalar)
        return {key: sum(items) / len(items) for key, items in values.items() if items}

    @staticmethod
    def _mean_list(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _pick_metric(metrics: dict[str, float], keys: tuple[str, ...], *fallbacks, default: float) -> float:
        for key in keys:
            if key in metrics:
                return metrics[key]
        for fallback in fallbacks:
            if isinstance(fallback, (int, float)):
                return float(fallback)
        return default

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _line(label: str, value: str | float | int, pad: int = 48) -> str:
        if isinstance(value, float):
            value = f"{value:.4f}"
        return f"{label:>{pad}}: {value}"

    @staticmethod
    def _first_float(value) -> float | None:
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                return float(value.detach().flatten()[0].item())
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    @classmethod
    def _range_values(cls, value) -> tuple[float | None, float | None]:
        if isinstance(value, torch.Tensor):
            flat = value.detach().flatten()
            if flat.numel() >= 2:
                return cls._first_float(flat[0]), cls._first_float(flat[1])
            scalar = cls._first_float(flat[0]) if flat.numel() == 1 else None
            return scalar, scalar
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return cls._first_float(value[0]), cls._first_float(value[1])
        scalar = cls._first_float(value)
        return scalar, scalar

    def _curriculum_metrics(self) -> dict[str, float]:
        env = self._unwrap_env()
        metrics: dict[str, float] = {}

        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if isinstance(terrain_levels, torch.Tensor) and terrain_levels.numel() > 0:
            terrain_levels = terrain_levels.float()
            metrics["Curriculum/terrain_level_mean"] = float(terrain_levels.mean().item())
            metrics["Curriculum/terrain_level_max"] = float(terrain_levels.max().item())

        curriculum_stats = getattr(env, "_waq_terrain_curriculum_stats", None)
        if isinstance(curriculum_stats, dict):
            for name, value in curriculum_stats.items():
                scalar = self._scalar(value)
                if scalar is not None:
                    metrics[f"Curriculum/terrain_{name}"] = scalar

        command_manager = getattr(env, "command_manager", None)
        if command_manager is not None:
            try:
                command_term = command_manager.get_term("base_velocity")
            except (AttributeError, KeyError):
                command_term = None
            ranges = getattr(getattr(command_term, "cfg", None), "ranges", None)
            if ranges is not None:
                for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                    low, high = self._range_values(getattr(ranges, name, None))
                    if low is not None:
                        metrics[f"Curriculum/{name}_min"] = low
                    if high is not None:
                        metrics[f"Curriculum/{name}_max"] = high

        return metrics

    def _log(
        self,
        iteration: int,
        total_iterations: int,
        learning_start_iteration: int,
        collection_time: float,
        learn_time: float,
        losses: dict[str, float],
        ep_infos: list[dict],
        rewbuffer: deque[float],
        lenbuffer: deque[float],
    ):
        self.tot_timesteps += self.cfg.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        fps = int(self.cfg.num_steps_per_env * self.env.num_envs / (collection_time + learn_time))
        episode_metrics = self._mean_episode_infos(ep_infos)
        rollout_stats = self._last_rollout_stats
        done_rate = float(rollout_stats.get("done_rate", 0.0) or 0.0)
        timeout_rate = float(rollout_stats.get("timeout_rate", 0.0) or 0.0)
        action_abs = float(rollout_stats.get("action_abs", 0.0) or 0.0)
        action_mean_abs = float(rollout_stats.get("action_mean_abs", 0.0) or 0.0)
        action_noise_abs = float(rollout_stats.get("action_noise_abs", 0.0) or 0.0)
        command_x = float(rollout_stats.get("command_x", 0.0) or 0.0)
        velocity_x = float(rollout_stats.get("velocity_x", 0.0) or 0.0)
        mean_reward = self._pick_metric(
            episode_metrics,
            ("Train/mean_reward", "Episode/reward", "Episode_Reward/total"),
            rollout_stats.get("mean_completed_reward"),
            statistics.mean(rewbuffer) if len(rewbuffer) > 0 else None,
            default=0.0,
        )
        mean_length = self._pick_metric(
            episode_metrics,
            ("Train/mean_episode_length", "Episode/length", "Episode_Length/episode"),
            rollout_stats.get("mean_completed_episode_length"),
            statistics.mean(lenbuffer) if len(lenbuffer) > 0 else None,
            default=0.0,
        )
        has_completed_episode = (
            rollout_stats.get("mean_completed_episode_length") is not None
            or len(lenbuffer) > 0
            or any(
                key in episode_metrics
                for key in ("Train/mean_episode_length", "Episode/length", "Episode_Length/episode")
            )
        )
        completed_this_call = iteration - learning_start_iteration + 1
        eta = self.tot_time / max(completed_this_call, 1) * max(total_iterations - iteration - 1, 0)
        run_name = self.cfg.run_name or (os.path.basename(self.log_dir) if self.log_dir is not None else "")
        mean_action_std = torch.exp(self.policy.log_std).mean().item()
        reward_metrics = self._last_reward_metrics
        curriculum_metrics = self._curriculum_metrics()
        termination_metrics = {
            key.removeprefix("Episode_Termination/"): value
            for key, value in episode_metrics.items()
            if key.startswith("Episode_Termination/")
        }

        if self.writer is not None:
            self.writer.add_scalar("Loss/value_function", losses["value"], iteration)
            self.writer.add_scalar("Loss/surrogate", losses["surrogate"], iteration)
            self.writer.add_scalar("Loss/entropy", -losses["entropy"], iteration)
            self.writer.add_scalar("Loss/estimator_velocity", losses["estimator"], iteration)
            self.writer.add_scalar("Loss/estimator_terrain", losses["terrain"], iteration)
            self.writer.add_scalar("Loss/estimator_kl", losses["kl"], iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg_cfg.learning_rate, iteration)
            self.writer.add_scalar("Policy/mean_noise_std", mean_action_std, iteration)
            self.writer.add_scalar("Perf/total_fps", fps, iteration)
            self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
            self.writer.add_scalar("Rollout/done_rate", done_rate, iteration)
            self.writer.add_scalar("Rollout/timeout_rate", timeout_rate, iteration)
            self.writer.add_scalar("Rollout/action_abs", action_abs, iteration)
            self.writer.add_scalar("Rollout/action_mean_abs", action_mean_abs, iteration)
            self.writer.add_scalar("Rollout/action_noise_abs", action_noise_abs, iteration)
            self.writer.add_scalar("Rollout/command_x", command_x, iteration)
            self.writer.add_scalar("Rollout/velocity_x", velocity_x, iteration)
            if len(rewbuffer) > 0:
                self.writer.add_scalar("Train/mean_reward", mean_reward, iteration)
                self.writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
            for key, value in episode_metrics.items():
                self.writer.add_scalar(key, value, iteration)
            for key, value in reward_metrics.items():
                self.writer.add_scalar(key, value, iteration)
            for key, value in curriculum_metrics.items():
                self.writer.add_scalar(key, value, iteration)

        log_lines = [
            f"{'#' * 80}",
            f"{f'Learning iteration {iteration + 1}/{total_iterations}':^80}",
            "",
        ]
        if run_name:
            log_lines.append(self._line("Run name", run_name))
        log_lines.extend(
            [
                self._line("Total steps", str(self.tot_timesteps)),
                self._line("Steps per second", str(fps)),
                self._line("Collection time", f"{collection_time:.3f}s"),
                self._line("Learning time", f"{learn_time:.3f}s"),
                self._line("Mean value loss", losses["value"]),
                self._line("Mean surrogate loss", losses["surrogate"]),
                self._line("Mean entropy loss", -losses["entropy"]),
                self._line("Mean estimator_velocity loss", losses["estimator"]),
                self._line("Mean estimator_terrain loss", losses["terrain"]),
                self._line("Mean estimator_kl loss", losses["kl"]),
                self._line("Mean reward", f"{mean_reward:.2f}" if has_completed_episode else "n/a"),
                self._line("Mean episode length", f"{mean_length:.2f}" if has_completed_episode else "n/a"),
                self._line("Mean action std", f"{mean_action_std:.2f}"),
                self._line("Rollout/done_rate", f"{done_rate:.4f}"),
                self._line("Rollout/timeout_rate", f"{timeout_rate:.4f}"),
                self._line("Rollout/action |mean|", f"{action_mean_abs:.4f}"),
                self._line("Rollout/action |sample|", f"{action_abs:.4f}"),
                self._line("Rollout/action |noise|", f"{action_noise_abs:.4f}"),
                self._line("Rollout/cmd_x vs vel_x", f"{command_x:.3f} / {velocity_x:.3f}"),
            ]
        )
        if reward_metrics:
            log_lines.extend([f"{'-' * 80}", f"{'Reward terms (rollout mean)':>48}:"])
            for key, value in reward_metrics.items():
                log_lines.append(self._line(key.rsplit("/", 1)[-1], value))
        if termination_metrics:
            log_lines.extend([f"{'-' * 80}", f"{'Termination terms (episode mean)':>48}:"])
            for key, value in termination_metrics.items():
                log_lines.append(self._line(key, value))
        if curriculum_metrics:
            log_lines.extend(
                [
                    f"{'-' * 80}",
                    self._line(
                        "Terrain level mean/max",
                        (
                            f"{curriculum_metrics.get('Curriculum/terrain_level_mean', 0.0):.2f}/"
                            f"{curriculum_metrics.get('Curriculum/terrain_level_max', 0.0):.0f}"
                        ),
                    ),
                    self._line(
                        "Terrain allowed/up/down",
                        (
                            f"{curriculum_metrics.get('Curriculum/terrain_allowed_max_level', 0.0):.0f}/"
                            f"{curriculum_metrics.get('Curriculum/terrain_move_up_rate', 0.0):.4f}/"
                            f"{curriculum_metrics.get('Curriculum/terrain_move_down_rate', 0.0):.4f}"
                        ),
                    ),
                    self._line(
                        "Terrain success/streak",
                        (
                            f"{curriculum_metrics.get('Curriculum/terrain_success_rate', 0.0):.4f}/"
                            f"{curriculum_metrics.get('Curriculum/terrain_success_streak_mean', 0.0):.2f}"
                        ),
                    ),
                    self._line(
                        "Command lin x range",
                        (
                            f"[{curriculum_metrics.get('Curriculum/lin_vel_x_min', 0.0):.2f}, "
                            f"{curriculum_metrics.get('Curriculum/lin_vel_x_max', 0.0):.2f}]"
                        ),
                    ),
                    self._line(
                        "Command lin y range",
                        (
                            f"[{curriculum_metrics.get('Curriculum/lin_vel_y_min', 0.0):.2f}, "
                            f"{curriculum_metrics.get('Curriculum/lin_vel_y_max', 0.0):.2f}]"
                        ),
                    ),
                    self._line(
                        "Command yaw range",
                        (
                            f"[{curriculum_metrics.get('Curriculum/ang_vel_z_min', 0.0):.2f}, "
                            f"{curriculum_metrics.get('Curriculum/ang_vel_z_max', 0.0):.2f}]"
                        ),
                    ),
                ]
            )
        log_lines.extend(
            [
                f"{'-' * 80}",
                self._line("Iteration time", f"{collection_time + learn_time:.2f}s"),
                self._line("Time elapsed", self._format_time(self.tot_time)),
                self._line("ETA", self._format_time(eta)),
            ]
        )
        print("\n".join(log_lines))

        if has_completed_episode and done_rate > 0.01 and mean_length <= 2.0 and iteration < 20:
            print(
                "[WAQ warning] episode_len <= 2.0: environments are resetting almost immediately. "
                "Check terminations before trusting policy learning."
            )

    def save(self, path: str, infos: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "cfg": self.cfg.to_dict(),
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True, curriculum_step_offset: int | None = None):
        loaded_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        env = self._unwrap_env()
        if curriculum_step_offset is None:
            curriculum_step_offset = self.current_learning_iteration * self.cfg.num_steps_per_env
        env._waq_curriculum_step_offset = int(curriculum_step_offset)
        return loaded_dict.get("infos")

    def get_inference_policy(self):
        self.policy.eval()
        return self.policy.act_inference
