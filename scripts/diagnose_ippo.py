from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.ippo import ActorCritic
from envs import WarehouseConfig, WarehouseEnv
from utils import load_config, load_warehouse_config, progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default=None, choices=["greedy", "sample", "both"])
    parser.add_argument("--sample-candidates", type=int, default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    env_config = load_warehouse_config(config_path)
    evaluation = config.get("evaluation", {})
    outputs = config.get("outputs", {})
    device = _select_device(args.device)

    model_path = args.model or str(Path(outputs.get("models_dir", "outputs/models")) / "ippo_actor_critic.pt")
    checkpoint = torch.load(PROJECT_ROOT / model_path, map_location=device)
    model = ActorCritic(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_size"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    modes = _evaluation_modes(args.mode or str(evaluation.get("policy_mode", "both")))

    rows = []
    episodes = int(evaluation.get("episodes", 5))
    for mode in progress(modes, total=len(modes), desc="evaluate modes", unit="mode"):
        sample_candidates = _candidate_count(mode, args.sample_candidates, evaluation)
        for episode in progress(
            range(1, episodes + 1),
            total=episodes,
            desc=f"evaluate {mode}",
            unit="episode",
        ):
            rows.append(_evaluate_episode(episode, env_config, model, device, mode, sample_candidates))

    logs_dir = PROJECT_ROOT / outputs.get("logs_dir", "outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / "evaluate_ippo.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        avg_completion = sum(row["completion_rate"] for row in mode_rows) / len(mode_rows)
        avg_reward = sum(row["episode_reward"] for row in mode_rows) / len(mode_rows)
        print(f"evaluate ippo {mode}: avg_completion={avg_completion:.3f}, avg_reward={avg_reward:.2f}")
    print(f"evaluation log saved: {output_path}")


def _evaluate_episode(
    episode: int,
    env_config: WarehouseConfig,
    model: ActorCritic,
    device: torch.device,
    mode: str,
    sample_candidates: int,
) -> dict[str, float | int | str]:
    candidates = []
    for candidate in progress(
        range(1, sample_candidates + 1),
        total=sample_candidates,
        desc=f"episode {episode} candidates",
        unit="candidate",
        leave=False,
    ):
        env = WarehouseEnv(env_config)
        env.reset_count = episode - 1
        seed = 10_000 + episode * 100 + candidate
        candidates.append(_run_episode(episode, candidate, env, model, device, mode, seed))
    return max(candidates, key=_selection_key)


def _candidate_count(mode: str, requested: int | None, evaluation: dict) -> int:
    if mode == "greedy":
        return 1
    return max(1, int(requested or evaluation.get("sample_candidates", 1)))


def _evaluation_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["greedy", "sample"]
    if mode in {"greedy", "sample"}:
        return [mode]
    raise ValueError("policy_mode must be 'greedy', 'sample', or 'both'")


def _run_episode(
    episode: int,
    candidate: int,
    env: WarehouseEnv,
    model: ActorCritic,
    device: torch.device,
    mode: str,
    seed: int,
) -> dict[str, float | int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    observations = env.reset()
    done = False
    total_reward = 0.0
    info = {}
    while not done:
        obs_array = np.asarray(observations, dtype=np.float32)
        if mode == "sample":
            sampled_actions, _, _ = model.act(obs_array, device)
            actions = sampled_actions.tolist()
        else:
            actions = model.greedy_actions(obs_array, device)
        actions = env.shield_actions(actions)
        observations, rewards, done, info = env.step(actions)
        total_reward += float(sum(rewards))

    completed = int(info["completed_tasks"])
    total_tasks = len(env.config.tasks)
    completed_priority = float(info.get("completed_priority", 0.0))
    total_priority = float(info.get("total_priority", 0.0))
    return {
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
        "collision_count": int(info["collision_count"]),
        "timeout_count": int(info["timeout_count"]),
        "total_battery": round(float(info.get("total_battery", 0.0)), 4),
        "steps": env.step_count,
    }


def _selection_key(row: dict[str, float | int]) -> tuple[float, float, float, float, float]:
    return (
        float(row["completion_rate"]),
        float(row["priority_completion_rate"]),
        -float(row["steps"]),
        float(row["episode_reward"]),
        -float(row["collision_count"]),
    )


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
