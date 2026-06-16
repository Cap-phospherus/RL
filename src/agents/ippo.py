from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from envs import Action


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(observations)
        values = self.critic(observations).squeeze(-1)
        return logits, values

    def masked_logits(self, observations: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        mask = action_mask_from_observations(observations, logits.shape[-1])
        return logits.masked_fill(~mask, -1e9)

    @torch.no_grad()
    def act(self, observations: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        logits, values = self(obs_tensor)
        logits = self.masked_logits(obs_tensor, logits)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def greedy_actions(self, observations: np.ndarray, device: torch.device) -> list[int]:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        logits, _ = self(obs_tensor)
        logits = self.masked_logits(obs_tensor, logits)
        return torch.argmax(logits, dim=-1).cpu().tolist()

    @torch.no_grad()
    def log_probs_for_actions(
        self,
        observations: np.ndarray,
        actions: list[int] | np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        logits, _ = self(obs_tensor)
        logits = self.masked_logits(obs_tensor, logits)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action_tensor).cpu().numpy()


def action_mask_from_observations(observations: torch.Tensor, action_dim: int) -> torch.Tensor:
    mask = torch.zeros((observations.shape[0], action_dim), dtype=torch.bool, device=observations.device)
    if action_dim <= int(Action.RIGHT):
        mask[:] = True
        return mask

    carrying = observations[:, 5] > 0.5
    active_priority = observations[:, 14] > 1e-4
    available_tasks = observations[:, 15] > 1e-4
    low_battery = observations[:, 16] > 0.5
    has_task = carrying | active_priority

    for action in [Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
        mask[:, int(action)] = has_task

    mask[:, int(Action.SELECT_NEAREST_TASK)] = available_tasks & ~has_task
    mask[:, int(Action.SELECT_PRIORITY_TASK)] = available_tasks & ~has_task
    mask[:, int(Action.GO_CHARGE)] = low_battery & ~carrying

    fallback = ~mask.any(dim=1)
    mask[fallback, int(Action.STAY)] = True
    return mask


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def build_batch(
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    log_probs: list[np.ndarray],
    rewards: list[np.ndarray],
    values: list[np.ndarray],
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> RolloutBatch:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards_array, dtype=np.float32)
    last_advantage = np.zeros(rewards_array.shape[1], dtype=np.float32)

    for step in reversed(range(len(rewards_array))):
        next_value = values_array[step + 1] if step + 1 < len(values_array) else 0.0
        delta = rewards_array[step] + gamma * next_value - values_array[step]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values_array
    flat_observations = np.asarray(observations, dtype=np.float32).reshape(-1, np.asarray(observations[0]).shape[-1])
    flat_actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    flat_log_probs = np.asarray(log_probs, dtype=np.float32).reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

    return RolloutBatch(
        observations=torch.as_tensor(flat_observations, dtype=torch.float32, device=device),
        actions=torch.as_tensor(flat_actions, dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(flat_log_probs, dtype=torch.float32, device=device),
        returns=torch.as_tensor(flat_returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(flat_advantages, dtype=torch.float32, device=device),
    )
