"""Rollout storage for DreamWaQ PPO."""

from __future__ import annotations

import torch


class DreamWaQRolloutStorage:
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        history_dim: int,
        critic_obs_dim: int,
        estimator_target_dim: int,
        action_dim: int,
        device: str,
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.step = 0

        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.histories = torch.zeros(num_steps, num_envs, history_dim, device=device)
        self.critic_observations = torch.zeros(num_steps, num_envs, critic_obs_dim, device=device)
        self.estimator_targets = torch.zeros(num_steps, num_envs, estimator_target_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.dones = torch.zeros(num_steps, num_envs, 1, device=device, dtype=torch.bool)
        self.values = torch.zeros(num_steps, num_envs, 1, device=device)
        self.returns = torch.zeros(num_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_steps, num_envs, 1, device=device)
        self.mu = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.sigma = torch.zeros(num_steps, num_envs, action_dim, device=device)

    def add(
        self,
        *,
        observations: torch.Tensor,
        histories: torch.Tensor,
        critic_observations: torch.Tensor,
        estimator_targets: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        action_mean: torch.Tensor,
        action_sigma: torch.Tensor,
    ):
        if self.step >= self.num_steps:
            raise RuntimeError("Rollout buffer overflow.")
        self.observations[self.step].copy_(observations)
        self.histories[self.step].copy_(histories)
        self.critic_observations[self.step].copy_(critic_observations)
        self.estimator_targets[self.step].copy_(estimator_targets)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1).bool())
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(action_mean)
        self.sigma[self.step].copy_(action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            next_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1.0e-8)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches
        usable_batch_size = mini_batch_size * num_mini_batches

        observations = self.observations.flatten(0, 1)
        histories = self.histories.flatten(0, 1)
        critic_observations = self.critic_observations.flatten(0, 1)
        estimator_targets = self.estimator_targets.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)[:usable_batch_size]
            for index in range(num_mini_batches):
                batch_idx = indices[index * mini_batch_size : (index + 1) * mini_batch_size]
                yield (
                    observations[batch_idx],
                    histories[batch_idx],
                    critic_observations[batch_idx],
                    estimator_targets[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
