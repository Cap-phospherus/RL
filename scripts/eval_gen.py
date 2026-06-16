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


PolicyRow = dict[str, float | int | str]
StartTuple = tuple[tuple[int, int], ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policy generalization on random robot starts unseen during training."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sample-candidates", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--completion-threshold", type=float, default=None)
    parser.add_argument("--priority-threshold", type=float, default=None)
    parser.add_argument("--max-mean-collisions", type=float, default=None)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    generalization = config.get("generalization", {})
    episodes = int(args.episodes if args.episodes is not None else generalization.get("episodes", 100))
    sample_candidates = int(
        args.sample_candidates if args.sample_candidates is not None else generalization.get("sample_candidates", 16)
    )
    seed_base = int(args.seed_base if args.seed_base is not None else generalization.get("seed_base", 1000))
    completion_threshold = float(
        args.completion_threshold
        if args.completion_threshold is not None
        else generalization.get("completion_threshold", 0.95)
    )
    priority_threshold = float(
        args.priority_threshold
        if args.priority_threshold is not None
        else generalization.get("priority_threshold", 0.95)
    )
    max_mean_collisions = float(
        args.max_mean_collisions
        if args.max_mean_collisions is not None
        else generalization.get("max_mean_collisions", 60.0)
    )
    output_prefix = str(args.output_prefix or generalization.get("output_prefix", "generalization_random_start"))
    base_env_config = load_warehouse_config(config_path)
    random_env_config = replace(
        base_env_config,
        deterministic_resets=False,
        randomize_starts=True,
        start_scenarios=None,
        robot_starts=None,
    )
    train_start_set = _training_start_set(base_env_config)
    device = _select_device(args.device)
    outputs = config.get("outputs", {})
    model_path = args.model or str(Path(outputs.get("models_dir", "outputs/models")) / "ippo_actor_critic.pt")
    model = _load_model(PROJECT_ROOT / model_path, device)

    logs_dir = PROJECT_ROOT / outputs.get("logs_dir", "outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows_path = logs_dir / f"{output_prefix}_rows.csv"
    starts_path = logs_dir / f"{output_prefix}_starts.csv"
    summary_path = logs_dir / f"{output_prefix}_summary.csv"
    verdict_path = logs_dir / f"{output_prefix}_verdict.txt"

    rows: list[PolicyRow] = []
    starts: list[dict[str, str | int | bool]] = []
    seen_random_starts: set[StartTuple] = set()

    for episode in progress(range(1, episodes + 1), total=episodes, desc="generalization", unit="episode"):
        episode_seed = seed_base + episode
        start_positions = _sample_unseen_start(
            random_env_config,
            episode_seed,
            train_start_set,
            seen_random_starts,
        )
        starts.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "start_positions": _format_starts(start_positions),
                "seen_in_training": start_positions in train_start_set,
            }
        )
        rows.append(_run_random(random_env_config, episode, episode_seed, start_positions))
        rows.append(_run_greedy(random_env_config, episode, episode_seed, start_positions, shield=False))
        rows.append(_run_greedy(random_env_config, episode, episode_seed, start_positions, shield=True))
        rows.append(_run_ippo(random_env_config, episode, episode_seed, start_positions, model, device, "greedy", 1))
        rows.append(
            _run_ippo(
                random_env_config,
                episode,
                episode_seed,
                start_positions,
                model,
                device,
                "sample",
                sample_candidates,
            )
        )

        # Keep partial files useful if a long run is interrupted.
        _write_csv(rows_path, rows)
        _write_csv(starts_path, starts)
        _write_csv(summary_path, _summaries(rows))

    summary_rows = _summaries(rows)
    verdict = _build_verdict(
        summary_rows,
        episodes=episodes,
        sample_candidates=sample_candidates,
        completion_threshold=completion_threshold,
        priority_threshold=priority_threshold,
        max_mean_collisions=max_mean_collisions,
    )
    verdict_path.write_text(verdict, encoding="utf-8")

    print(verdict)
    print(f"rows saved: {rows_path}")
    print(f"starts saved: {starts_path}")
    print(f"summary saved: {summary_path}")
    print(f"verdict saved: {verdict_path}")


def _training_start_set(config: WarehouseConfig) -> set[StartTuple]:
    if not config.start_scenarios:
        return set()
    return {tuple(tuple(position) for position in scenario) for scenario in config.start_scenarios}


def _sample_unseen_start(
    config: WarehouseConfig,
    seed: int,
    train_start_set: set[StartTuple],
    seen_random_starts: set[StartTuple],
) -> StartTuple:
    # Deterministically resample until the start is outside the training set and not duplicated in this run.
    for offset in range(10_000):
        env = WarehouseEnv(config)
        env.rng.seed(seed + offset * 104_729)
        env.reset()
        start_positions = tuple(robot.position for robot in env.robots)
        if start_positions not in train_start_set and start_positions not in seen_random_starts:
            seen_random_starts.add(start_positions)
            return start_positions
    raise RuntimeError("Could not sample a unique unseen random start after 10000 attempts")


def _run_random(config: WarehouseConfig, episode: int, seed: int, starts: StartTuple) -> PolicyRow:
    env = _env_with_starts(config, starts)
    observations = env.reset()
    policy = RandomPolicy(env.action_size, seed=seed)
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    while not done:
        observations, rewards, done, info = env.step(policy.act(len(observations)))
        total_reward += float(sum(rewards))
    return _row("random", episode, 1, "single", env, total_reward, info)


def _run_greedy(config: WarehouseConfig, episode: int, seed: int, starts: StartTuple, shield: bool) -> PolicyRow:
    del seed
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
    seed: int,
    starts: StartTuple,
    model: ActorCritic,
    device: torch.device,
    mode: str,
    candidates: int,
) -> PolicyRow:
    candidate_rows = []
    for candidate in range(1, candidates + 1):
        torch.manual_seed(seed * 10_000 + candidate)
        np.random.seed(seed * 10_000 + candidate)
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
) -> PolicyRow:
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


def _selection_key(row: PolicyRow) -> tuple[float, float, float, float, float]:
    return (
        float(row["completion_rate"]),
        float(row["priority_completion_rate"]),
        -float(row["steps"]),
        float(row["episode_reward"]),
        -float(row["collision_count"]),
    )


def _summaries(rows: list[PolicyRow]) -> list[dict[str, float | int | str]]:
    summaries = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary: dict[str, float | int | str] = {"policy": policy, "episodes": len(policy_rows)}
        for key in [
            "episode_reward",
            "completion_rate",
            "priority_completion_rate",
            "collision_count",
            "timeout_count",
            "steps",
            "total_battery",
        ]:
            values = [float(row[key]) for row in policy_rows]
            mean_value = _mean(values)
            std_value = _std(values)
            ci_low, ci_high = _confidence_interval(values)
            summary[f"{key}_mean"] = round(mean_value, 4)
            summary[f"{key}_std"] = round(std_value, 4)
            summary[f"{key}_ci95_low"] = round(ci_low, 4)
            summary[f"{key}_ci95_high"] = round(ci_high, 4)
            summary[f"{key}_min"] = round(min(values), 4)
            summary[f"{key}_max"] = round(max(values), 4)
        summaries.append(summary)
    return summaries


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _confidence_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = _mean(values)
    if len(values) < 2:
        return mean_value, mean_value
    half_width = 1.96 * _std(values) / (len(values) ** 0.5)
    return mean_value - half_width, mean_value + half_width


def _build_verdict(
    summaries: list[dict[str, float | int | str]],
    episodes: int,
    sample_candidates: int,
    completion_threshold: float,
    priority_threshold: float,
    max_mean_collisions: float,
) -> str:
    lines = [
        "Random-start generalization evaluation",
        f"episodes={episodes}",
        f"sample_candidates={sample_candidates}",
        "All sampled starts are excluded from the training start_scenarios.",
        "",
    ]
    policy_map = {str(row["policy"]): row for row in summaries}
    ippo = policy_map.get("ippo_sample_shield")
    if ippo is None:
        lines.append("verdict=FAIL: missing ippo_sample_shield rows")
        return "\n".join(lines)

    completion_low = float(ippo["completion_rate_ci95_low"])
    priority_low = float(ippo["priority_completion_rate_ci95_low"])
    collision_mean = float(ippo["collision_count_mean"])
    passed = (
        completion_low >= completion_threshold
        and priority_low >= priority_threshold
        and collision_mean <= max_mean_collisions
    )
    lines.append(f"ippo_sample_shield completion_mean={ippo['completion_rate_mean']}")
    lines.append(f"ippo_sample_shield completion_ci95=[{ippo['completion_rate_ci95_low']}, {ippo['completion_rate_ci95_high']}]")
    lines.append(f"ippo_sample_shield priority_completion_mean={ippo['priority_completion_rate_mean']}")
    lines.append(
        "ippo_sample_shield priority_completion_ci95="
        f"[{ippo['priority_completion_rate_ci95_low']}, {ippo['priority_completion_rate_ci95_high']}]"
    )
    lines.append(f"ippo_sample_shield collision_mean={ippo['collision_count_mean']}")
    lines.append(
        "thresholds="
        f"completion_ci95_low>={completion_threshold}, "
        f"priority_ci95_low>={priority_threshold}, "
        f"collision_mean<={max_mean_collisions}"
    )
    lines.append(f"verdict={'PASS' if passed else 'NOT_PASS'}")
    lines.append("")
    lines.append(
        "Note: this is statistical evidence over finite unseen random starts, not a mathematical proof over all possible starts."
    )
    return "\n".join(lines)


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


def _format_starts(starts: StartTuple) -> str:
    return ";".join(f"({x},{y})" for x, y in starts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
