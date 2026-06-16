from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents import GreedyTaskPolicy, RandomPolicy
from agents.ippo import ActorCritic
from envs import WarehouseConfig, WarehouseEnv
from utils import load_config, load_warehouse_config, progress


Row = dict[str, float | int | str]
StartTuple = tuple[tuple[int, int], ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Random, Greedy, Greedy+shield, and IPPO on fixed training starts."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sample-candidates", type=int, default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    env_config = load_warehouse_config(config_path)
    evaluation = config.get("evaluation", {})
    outputs = config.get("outputs", {})
    sample_candidates = int(args.sample_candidates or evaluation.get("sample_candidates", 1))
    starts = _fixed_starts(env_config)
    device = _select_device(args.device)
    model_path = args.model or str(Path(outputs.get("models_dir", "outputs/models")) / "ippo_actor_critic.pt")
    model = _load_model(PROJECT_ROOT / model_path, device)

    logs_dir = PROJECT_ROOT / outputs.get("logs_dir", "outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows_path = logs_dir / "fixed_eval_rows.csv"
    summary_path = logs_dir / "fixed_eval_summary.csv"

    rows: list[Row] = []
    for episode, start_positions in progress(
        list(enumerate(starts, start=1)),
        total=len(starts),
        desc="evaluate fixed starts",
        unit="start",
    ):
        rows.append(_run_random(env_config, episode, start_positions))
        rows.append(_run_greedy(env_config, episode, start_positions, shield=False))
        rows.append(_run_greedy(env_config, episode, start_positions, shield=True))
        rows.append(_run_ippo(env_config, episode, start_positions, model, device, mode="greedy", candidates=1))
        rows.append(
            _run_ippo(
                env_config,
                episode,
                start_positions,
                model,
                device,
                mode="sample",
                candidates=sample_candidates,
            )
        )
        _write_csv(rows_path, rows)
        _write_csv(summary_path, _summaries(rows))

    summaries = _summaries(rows)
    _write_csv(rows_path, rows)
    _write_csv(summary_path, summaries)
    for row in summaries:
        print(
            f"{row['policy']}: completion_rate={row['completion_rate']}, "
            f"priority_completion={row['priority_completion_rate']}, "
            f"collisions={row['collision_count']}, reward={row['episode_reward']}"
        )
    print(f"fixed-start rows saved: {rows_path}")
    print(f"fixed-start summary saved: {summary_path}")


def _fixed_starts(config: WarehouseConfig) -> list[StartTuple]:
    if config.start_scenarios:
        return [tuple(tuple(position) for position in scenario) for scenario in config.start_scenarios]
    if config.robot_starts:
        return [tuple(tuple(position) for position in config.robot_starts)]
    env = WarehouseEnv(config)
    env.reset()
    return [tuple(robot.position for robot in env.robots)]


def _run_random(config: WarehouseConfig, episode: int, starts: StartTuple) -> Row:
    env = _env_with_starts(config, starts)
    observations = env.reset()
    policy = RandomPolicy(env.action_size, seed=config.seed + episode)
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    while not done:
        observations, rewards, done, info = env.step(policy.act(len(observations)))
        total_reward += float(sum(rewards))
    return _row("random", episode, 1, "single", env, total_reward, info)


def _run_greedy(config: WarehouseConfig, episode: int, starts: StartTuple, shield: bool) -> Row:
    env = _env_with_starts(config, starts)
    env.reset()
    policy = GreedyTaskPolicy()
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    while not done:
        actions = policy.act(env)
        if shield:
            actions = env.shield_actions(actions)
        _, rewards, done, info = env.step(actions)
        total_reward += float(sum(rewards))
    return _row("greedy_shield" if shield else "greedy", episode, 1, "single", env, total_reward, info)


def _run_ippo(
    config: WarehouseConfig,
    episode: int,
    starts: StartTuple,
    model: ActorCritic,
    device: torch.device,
    mode: str,
    candidates: int,
) -> Row:
    candidate_rows = []
    for candidate in range(1, candidates + 1):
        seed = 10_000 + episode * 100 + candidate
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = _env_with_starts(config, starts)
        observations = env.reset()
        total_reward = 0.0
        done = False
        info: dict[str, Any] = {}
        while not done:
            obs_array = np.asarray(observations, dtype=np.float32)
            if mode == "sample":
                actions, _, _ = model.act(obs_array, device)
                actions = actions.tolist()
            else:
                actions = model.greedy_actions(obs_array, device)
            actions = env.shield_actions(actions)
            observations, rewards, done, info = env.step(actions)
            total_reward += float(sum(rewards))
        candidate_rows.append(_row(f"ippo_{mode}_shield", episode, candidate, mode, env, total_reward, info))
    return max(candidate_rows, key=_selection_key)


def _env_with_starts(config: WarehouseConfig, starts: StartTuple) -> WarehouseEnv:
    return WarehouseEnv(
        replace(
            config,
            deterministic_resets=True,
            randomize_starts=False,
            start_scenarios=None,
            robot_starts=list(starts),
        )
    )


def _row(
    policy: str,
    episode: int,
    candidate: int,
    mode: str,
    env: WarehouseEnv,
    total_reward: float,
    info: dict[str, Any],
) -> Row:
    completed = int(info.get("completed_tasks", 0))
    total_tasks = len(env.config.tasks)
    completed_priority = float(info.get("completed_priority", 0.0))
    total_priority = float(info.get("total_priority", 0.0))
    return {
        "policy": policy,
        "episode": episode,
        "candidate": candidate,
        "mode": mode,
        "episode_reward": round(total_reward, 4),
        "completed_tasks": completed,
        "total_tasks": total_tasks,
        "completion_rate": round(completed / max(total_tasks, 1), 4),
        "completed_priority": round(completed_priority, 4),
        "total_priority": round(total_priority, 4),
        "priority_completion_rate": round(completed_priority / max(total_priority, 1.0), 4),
        "collision_count": int(info.get("collision_count", 0)),
        "timeout_count": int(info.get("timeout_count", 0)),
        "total_battery": round(float(info.get("total_battery", 0.0)), 4),
        "steps": env.step_count,
    }


def _summaries(rows: list[Row]) -> list[dict[str, float | int | str]]:
    summaries = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summaries.append(
            {
                "policy": policy,
                "episodes": len(policy_rows),
                "episode_reward": round(_mean(policy_rows, "episode_reward"), 4),
                "completion_rate": round(_mean(policy_rows, "completion_rate"), 4),
                "priority_completion_rate": round(_mean(policy_rows, "priority_completion_rate"), 4),
                "collision_count": round(_mean(policy_rows, "collision_count"), 4),
                "timeout_count": round(_mean(policy_rows, "timeout_count"), 4),
                "steps": round(_mean(policy_rows, "steps"), 4),
                "total_battery": round(_mean(policy_rows, "total_battery"), 4),
            }
        )
    return summaries


def _selection_key(row: Row) -> tuple[float, float, float, float, float]:
    return (
        float(row["completion_rate"]),
        float(row["priority_completion_rate"]),
        -float(row["steps"]),
        float(row["episode_reward"]),
        -float(row["collision_count"]),
    )


def _mean(rows: list[Row], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def _load_model(path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(path, map_location=device)
    model = ActorCritic(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_size"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
