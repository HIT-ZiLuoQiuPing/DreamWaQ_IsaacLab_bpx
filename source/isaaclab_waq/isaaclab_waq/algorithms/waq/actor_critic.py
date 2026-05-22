"""DreamWaQ actor-critic with a compact CENet encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims]
    for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(_activation(activation))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class EmpiricalNormalizer(nn.Module):
    """Running mean/std normalizer matching the lightweight RSL-RL behavior."""

    def __init__(self, size: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(m_2 / total_count, min=self.eps))
        self.count.copy_(total_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class DreamWaQActorCritic(nn.Module):
    """Actor-critic with DreamWaQ CENet context estimation."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_history_obs: int,
        num_critic_obs: int,
        num_estimator_target_obs: int,
        num_actions: int,
        *,
        init_noise_std: float,
        min_noise_std: float,
        max_noise_std: float,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        encoder_hidden_dims: list[int],
        decoder_hidden_dims: list[int],
        latent_dim: int,
        activation: str,
        normalize_observations: bool = True,
    ):
        super().__init__()
        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must not be empty.")
        if num_estimator_target_obs < 3:
            raise ValueError("DreamWaQ estimator target must contain at least 3D base linear velocity.")

        self.num_actor_obs = num_actor_obs
        self.num_history_obs = num_history_obs
        self.num_critic_obs = num_critic_obs
        self.num_estimator_target_obs = num_estimator_target_obs
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.context_dim = latent_dim + 3
        self.terrain_dim = num_estimator_target_obs - 3
        self.normalize_observations = normalize_observations
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

        if normalize_observations:
            self.actor_obs_normalizer = EmpiricalNormalizer(num_actor_obs)
            self.history_obs_normalizer = EmpiricalNormalizer(num_history_obs)
            self.critic_obs_normalizer = EmpiricalNormalizer(num_critic_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.history_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.encoder = _mlp(num_history_obs, encoder_hidden_dims[:-1], encoder_hidden_dims[-1], activation)
        encoder_out_dim = encoder_hidden_dims[-1]
        self.velocity_head = nn.Linear(encoder_out_dim, 3)
        self.latent_mean = nn.Linear(encoder_out_dim, latent_dim)
        self.latent_logvar = nn.Linear(encoder_out_dim, latent_dim)
        self.decoder = _mlp(latent_dim, decoder_hidden_dims, self.terrain_dim, activation)

        self.actor = _mlp(num_actor_obs + self.context_dim, actor_hidden_dims, num_actions, activation)
        self.critic = _mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        self.log_std = nn.Parameter(torch.log(torch.full((num_actions,), init_noise_std)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @torch.no_grad()
    def clamp_action_std(self):
        self.log_std.clamp_(min=math.log(self.min_noise_std), max=math.log(self.max_noise_std))

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been created yet.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been created yet.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been created yet.")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None):
        pass

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    @torch.no_grad()
    def update_normalization(self, observations: torch.Tensor, history: torch.Tensor, critic_observations: torch.Tensor):
        if not self.normalize_observations:
            return
        self.actor_obs_normalizer.update(observations)
        self.history_obs_normalizer.update(history)
        self.critic_obs_normalizer.update(critic_observations)

    def cenet_forward(self, history: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        history = self.history_obs_normalizer(history)
        encoded = self.encoder(history)
        velocity = self.velocity_head(encoded)
        latent_mean = self.latent_mean(encoded)
        latent_logvar = self.latent_logvar(encoded).clamp(-10.0, 4.0)

        latent_code = self._reparameterize(latent_mean, latent_logvar, sample)
        context = torch.cat((velocity, latent_code), dim=-1)
        terrain = self.decoder(latent_code)
        return {
            "context": context,
            "velocity": velocity,
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,
            "terrain": terrain,
        }

    def update_distribution(self, observations: torch.Tensor, history: torch.Tensor, sample_context: bool = True):
        observations = self.actor_obs_normalizer(observations)
        context = self.cenet_forward(history, sample=sample_context)["context"]
        mean = torch.tanh(self.actor(torch.cat((observations, context), dim=-1)))
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, history: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations, history, sample_context=False)
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been created yet.")
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been created yet.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        observations = self.actor_obs_normalizer(observations)
        context = self.cenet_forward(history, sample=False)["context"]
        return torch.tanh(self.actor(torch.cat((observations, context), dim=-1)))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(self.critic_obs_normalizer(critic_observations))

    def cenet_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.cenet_forward(history, sample=False)
        velocity_target = target[..., :3]
        terrain_target = target[..., 3:]
        velocity_loss = nn.functional.mse_loss(out["velocity"], velocity_target)
        if self.terrain_dim > 0:
            terrain_loss = nn.functional.mse_loss(out["terrain"], terrain_target)
        else:
            terrain_loss = target.new_zeros(())
        latent_mean = out["latent_mean"]
        latent_logvar = out["latent_logvar"]
        kl_loss = -0.5 * torch.mean(1.0 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        return {
            "estimator_velocity": velocity_loss,
            "estimator_terrain": terrain_loss,
            "kl": kl_loss,
        }
