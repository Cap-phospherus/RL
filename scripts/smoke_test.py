from __future__ import annotations

import csv
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents import RandomPolicy
from envs import WarehouseEnv
from evaluation import EpisodeMetrics
from utils import load_config, load_warehouse_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    env = WarehouseEnv(load_warehouse_config(config_path))
    observations = env.reset()
    policy = RandomPolicy(action_size=env.action_size, seed=env.config.seed)
    total_reward = 0.0
    done = False
    info = {}

    while not done:
        actions = policy.act(num_agents=len(observations))
        observations, rewards, done, info = env.step(actions)
        total_reward += sum(rewards)

    metrics = EpisodeMetrics(
        episode_reward=total_reward,
        completed_tasks=info["completed_tasks"],
        total_tasks=len(env.config.tasks),
        collision_count=info["collision_count"],
        timeout_count=info["timeout_count"],
        total_battery=info["total_battery"],
        steps=env.step_count,
    )

    outputs = config.get("outputs", {})
    logs_dir = PROJECT_ROOT / outputs.get("logs_dir", "outputs/smoke/logs")
    figures_dir = PROJECT_ROOT / outputs.get("figures_dir", "outputs/smoke/figures")
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "smoke_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode_reward",
                "completed_tasks",
                "total_tasks",
                "completion_rate",
                "collision_count",
                "timeout_count",
                "total_battery",
                "steps",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode_reward": round(metrics.episode_reward, 4),
                "completed_tasks": metrics.completed_tasks,
                "total_tasks": metrics.total_tasks,
                "completion_rate": round(metrics.completion_rate, 4),
                "collision_count": metrics.collision_count,
                "timeout_count": metrics.timeout_count,
                "total_battery": round(metrics.total_battery, 4),
                "steps": metrics.steps,
            }
        )

    env.save_snapshot(figures_dir / "smoke_final_state.png")
    print(
        "smoke ok: "
        f"completion_rate={metrics.completion_rate:.2f}, "
        f"collisions={metrics.collision_count}, "
        f"reward={metrics.episode_reward:.2f}"
    )


if __name__ == "__main__":
    main()
