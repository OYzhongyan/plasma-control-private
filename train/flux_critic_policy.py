"""SB3 policy with the baseline vector actor and an independent flux critic.

Only the value branch consumes the four normalized flux frames. Constructing
the baseline actor/value MLP first preserves baseline actor initialization and
the random state used by the first sampled action. The temporary value branch
is removed before rebuilding the optimizer around the final parameters.
"""
from typing import Any

from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule, TensorDict


class ActorObsExtractor(BaseFeaturesExtractor):
    """Parameter-free selector retaining the original 812 actor inputs."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=812)

    def forward(self, observations: TensorDict) -> torch.Tensor:
        return observations["actor_obs"]


class FluxCriticNet(nn.Module):
    """Shared per-frame encoder and ordered, masked history value network."""

    def __init__(self):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(5184, 128), nn.ReLU(),
        )
        self.aux_encoder = nn.Sequential(nn.Linear(812, 128), nn.ReLU())
        self.value_head = nn.Sequential(
            nn.Linear(644, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, observations: TensorDict) -> torch.Tensor:
        frames = observations["flux_history"].float()
        mask = observations["flux_mask"].float()
        batch_size = frames.shape[0]
        embeddings = self.frame_encoder(frames.reshape(-1, 1, 66, 65))
        embeddings = embeddings.reshape(batch_size, 4, 128) * mask.unsqueeze(-1)
        auxiliary = self.aux_encoder(observations["actor_obs"].float())
        return self.value_head(torch.cat((embeddings.flatten(1), auxiliary, mask), dim=1))


class FluxCriticPolicy(ActorCriticPolicy):
    """Baseline Gaussian actor with an isolated four-frame CNN value branch.

    The default actor matches the historical 812 -> 256 -> 256 -> 7 Tanh
    policy with initial log standard deviation -1. Standard SB3 policy kwargs
    (including optimizer settings) remain supported. Observation shapes and
    the parameter-free actor extractor are fixed by this experiment's schema.
    Action-only ``predict``/``get_distribution`` inherit SB3's actor path and
    never execute ``flux_critic``.
    """

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Space,
                 lr_schedule: Schedule, **kwargs: Any):
        expected = {"actor_obs": (812,), "flux_history": (4, 66, 65), "flux_mask": (4,)}
        if not isinstance(observation_space, spaces.Dict) or any(
            key not in observation_space.spaces or observation_space[key].shape != shape
            for key, shape in expected.items()
        ):
            raise ValueError(f"FluxCriticPolicy requires Dict observation shapes {expected}")
        if kwargs.get("features_extractor_class", ActorObsExtractor) is not ActorObsExtractor:
            raise ValueError("FluxCriticPolicy requires the parameter-free ActorObsExtractor")
        kwargs["features_extractor_class"] = ActorObsExtractor
        kwargs.setdefault("net_arch", dict(pi=[256, 256], vf=[256, 256]))
        kwargs.setdefault("activation_fn", nn.Tanh)
        kwargs.setdefault("log_std_init", -1.)
        kwargs.setdefault("normalize_images", False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # No residual baseline value parameters may enter training/checkpoints.
        self.mlp_extractor.value_net = nn.Identity()
        self.value_net = nn.Identity()
        # CPU construction is the same as SB3's own construction. PPO moves
        # the complete policy to its target device after this method returns.
        with torch.random.fork_rng(devices=[]):
            self.flux_critic = FluxCriticNet().to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: TensorDict, deterministic: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return (actions.reshape((-1, *self.action_space.shape)),
                self.predict_values(obs), log_prob)

    def evaluate_actions(self, obs: TensorDict, actions: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        distribution = self.get_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        return self.flux_critic(obs)
