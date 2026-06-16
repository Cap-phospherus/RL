from __future__ import annotations

import random


class RandomPolicy:
    def __init__(self, action_size: int, seed: int = 0) -> None:
        self.action_size = action_size
        self.rng = random.Random(seed)

    def act(self, num_agents: int) -> list[int]:
        return [self.rng.randrange(self.action_size) for _ in range(num_agents)]
