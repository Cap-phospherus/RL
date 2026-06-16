from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_reward: float
    completed_tasks: int
    total_tasks: int
    collision_count: int
    timeout_count: int
    total_battery: float
    steps: int

    @property
    def completion_rate(self) -> float:
        return self.completed_tasks / max(self.total_tasks, 1)
