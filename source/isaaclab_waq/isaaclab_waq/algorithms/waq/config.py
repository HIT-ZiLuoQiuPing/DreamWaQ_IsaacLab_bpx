"""Configuration for the DreamWaQ-style BPX trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DreamWaQPolicyCfg:
    init_noise_std: float = 0.55
    min_noise_std: float = 0.15
    max_noise_std: float = 0.55
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 512])
    latent_dim: int = 24
    activation: str = "elu"
    normalize_observations: bool = True


@dataclass
class DreamWaQPpoCfg:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float | None = 0.01
    max_grad_norm: float = 1.0
    estimator_loss_coef: float = 1.0
    reconstruction_loss_coef: float = 0.2
    kl_loss_coef: float = 1.0e-3


@dataclass
class DreamWaQConfig:
    num_steps_per_env: int = 12
    max_iterations: int = 50000
    save_interval: int = 200
    experiment_name: str = "bpx_waq_rough"
    run_name: str = ""
    seed: int = 42
    clip_actions: float | None = 1.0
    policy: DreamWaQPolicyCfg = field(default_factory=DreamWaQPolicyCfg)
    algorithm: DreamWaQPpoCfg = field(default_factory=DreamWaQPpoCfg)

    def to_dict(self) -> dict:
        return asdict(self)
