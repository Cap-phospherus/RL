from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents import GreedyTaskPolicy
from agents.ippo import ActorCritic, build_batch
from envs import WarehouseConfig, WarehouseEnv
from utils import load_config, load_warehouse_config, progress, progress_write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    env_config = load_warehouse_config(config_path)
    training = config.get("training", {})
    outputs = config.get("outputs", {})

    device = _select_device(args.device)
    torch.manual_seed(env_config.seed)
    np.random.seed(env_config.seed)

    episodes = int(args.episodes or training.get("episodes", 200))
    stages = _build_curriculum({} if args.target_only else config, env_config, episodes)
    env = WarehouseEnv(stages[0]["env_config"])
    env.reset()
    model = ActorCritic(
        observation_dim=env.observation_dim,
        action_dim=env.action_size,
        hidden_dim=int(training.get("hidden_dim", 128)),
    ).to(device)
    if args.resume_model is not None:
        checkpoint = torch.load(PROJECT_ROOT / args.resume_model, map_location=device)
        if int(checkpoint["observation_dim"]) != env.observation_dim:
            raise ValueError("Resume model observation_dim does not match current environment")
        if int(checkpoint["action_size"]) != env.action_size:
            raise ValueError("Resume model action_size does not match current environment")
        model.load_state_dict(checkpoint["model_state_dict"])
        progress_write(f"resumed model: {PROJECT_ROOT / args.resume_model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 3e-4)))

    gamma = float(training.get("gamma", 0.99))
    gae_lambda = float(training.get("gae_lambda", 0.95))
    clip_ratio = float(training.get("clip_ratio", 0.2))
    update_epochs = int(training.get("update_epochs", 4))
    entropy_coef = float(training.get("entropy_coef", 0.01))
    value_coef = float(training.get("value_coef", 0.5))
    selection_eval_episodes = int(training.get("selection_eval_episodes", 1))
    selection_interval = int(training.get("selection_interval", 1))
    bc_episodes = int(training.get("behavior_clone_episodes", 0))
    bc_epochs = int(training.get("behavior_clone_epochs", 0))

    logs_dir = PROJECT_ROOT / outputs.get("logs_dir", "outputs/logs")
    figures_dir = PROJECT_ROOT / outputs.get("figures_dir", "outputs/figures")
    models_dir = PROJECT_ROOT / outputs.get("models_dir", "outputs/models")
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_model is None and bc_episodes > 0 and bc_epochs > 0:
        _behavior_clone_pretrain(
            model=model,
            env_config=env_config,
            device=device,
            episodes=bc_episodes,
            epochs=bc_epochs,
            learning_rate=float(training.get("behavior_clone_learning_rate", 1e-3)),
        )

    history = []
    best_score: tuple[float, int, float, int] | None = None
    best_episode = 0
    best_state_dict = None
    active_stage_index = -1
    for episode in progress(range(1, episodes + 1), total=episodes, desc="train ippo", unit="episode"):
        stage_index, stage = _stage_for_episode(stages, episode)
        if stage_index != active_stage_index:
            env = WarehouseEnv(stage["env_config"])
            active_stage_index = stage_index
            progress_write(f"stage={stage['name']} episodes={stage['start']}-{stage['end']}")
        rollout = _collect_episode(env, model, device)
        batch = build_batch(
            rollout["observations"],
            rollout["actions"],
            rollout["log_probs"],
            rollout["rewards"],
            rollout["values"],
            gamma,
            gae_lambda,
            device,
        )

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        for _ in range(update_epochs):
            logits, values = model(batch.observations)
            logits = model.masked_logits(batch.observations, logits)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(batch.actions)
            entropy = distribution.entropy().mean()
            ratio = torch.exp(new_log_probs - batch.old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch.returns)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            last_policy_loss = float(policy_loss.detach().cpu())
            last_value_loss = float(value_loss.detach().cpu())
            last_entropy = float(entropy.detach().cpu())

        row = {
            "episode": episode,
            "stage": stage["name"],
            "episode_reward": round(rollout["episode_reward"], 4),
            "completed_tasks": rollout["info"]["completed_tasks"],
            "total_tasks": len(env.config.tasks),
            "completion_rate": round(rollout["info"]["completed_tasks"] / max(len(env.config.tasks), 1), 4),
            "completed_priority": round(float(rollout["info"].get("completed_priority", 0.0)), 4),
            "total_priority": round(float(rollout["info"].get("total_priority", 0.0)), 4),
            "priority_completion_rate": round(
                float(rollout["info"].get("completed_priority", 0.0))
                / max(float(rollout["info"].get("total_priority", 0.0)), 1.0),
                4,
            ),
            "collision_count": rollout["info"]["collision_count"],
            "timeout_count": rollout["info"]["timeout_count"],
            "total_battery": round(float(rollout["info"].get("total_battery", 0.0)), 4),
            "steps": rollout["steps"],
            "policy_loss": round(last_policy_loss, 6),
            "value_loss": round(last_value_loss, 6),
            "entropy": round(last_entropy, 6),
        }
        history.append(row)
        score = _training_score(row)
        should_select = stage["select_checkpoint"] and (
            episode == episodes or episode % max(selection_interval, 1) == 0
        )
        if should_select and selection_eval_episodes > 1:
            score = _evaluate_selection_score(model, env_config, device, selection_eval_episodes)
        if stage["select_checkpoint"] and should_select and (best_score is None or score > best_score):
            best_score = score
            best_episode = episode
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if episode == 1 or episode == episodes or episode % max(1, episodes // 10) == 0:
            progress_write(
                f"episode={episode}/{episodes} "
                f"reward={row['episode_reward']} "
                f"completion={row['completion_rate']} "
                f"collisions={row['collision_count']}"
            )

    train_csv = logs_dir / "train_ippo.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    model_path = models_dir / "ippo_actor_critic.pt"
    final_model_path = models_dir / "ippo_actor_critic_final.pt"
    torch.save(
        {
            "model_state_dict": best_state_dict or model.state_dict(),
            "observation_dim": env.observation_dim,
            "action_size": env.action_size,
            "hidden_dim": int(training.get("hidden_dim", 128)),
            "config": config,
            "selected_episode": best_episode or episodes,
            "selection_metric": (
                "best priority_completion_rate, then completion_rate, fewer collisions/timeouts, "
                "then higher reward"
            ),
        },
        model_path,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_dim": env.observation_dim,
            "action_size": env.action_size,
            "hidden_dim": int(training.get("hidden_dim", 128)),
            "config": config,
            "selected_episode": episodes,
            "selection_metric": "final episode",
        },
        final_model_path,
    )
    _plot_training(history, figures_dir / "ippo_training_curves.png")
    progress_write(f"training log saved: {train_csv}")
    progress_write(f"best model saved: {model_path} episode={best_episode}")
    progress_write(f"final model saved: {final_model_path}")


def _build_curriculum(
    config: dict[str, Any],
    target_env_config: WarehouseConfig,
    total_episodes: int,
) -> list[dict[str, Any]]:
    raw_stages = config.get("curriculum", [])
    if not raw_stages:
        return [
            {
                "name": "target",
                "start": 1,
                "end": total_episodes,
                "env_config": target_env_config,
                "select_checkpoint": True,
            }
        ]

    stages = []
    start = 1
    for idx, raw_stage in enumerate(raw_stages):
        episodes = int(raw_stage.get("episodes", 0))
        if episodes <= 0:
            continue
        end = min(total_episodes, start + episodes - 1)
        overrides = raw_stage.get("env_overrides", {})
        stages.append(
            {
                "name": str(raw_stage.get("name", f"stage_{idx + 1}")),
                "start": start,
                "end": end,
                "env_config": _override_env_config(target_env_config, overrides),
                "select_checkpoint": bool(raw_stage.get("select_checkpoint", idx == len(raw_stages) - 1)),
            }
        )
        start = end + 1
        if start > total_episodes:
            break

    if start <= total_episodes:
        stages.append(
            {
                "name": "target",
                "start": start,
                "end": total_episodes,
                "env_config": target_env_config,
                "select_checkpoint": True,
            }
        )
    return stages


def _override_env_config(base: WarehouseConfig, overrides: dict[str, Any]) -> WarehouseConfig:
    valid_fields = set(WarehouseConfig.__dataclass_fields__)
    safe_overrides = {key: value for key, value in overrides.items() if key in valid_fields}
    return replace(base, **safe_overrides)


def _stage_for_episode(stages: list[dict[str, Any]], episode: int) -> tuple[int, dict[str, Any]]:
    for idx, stage in enumerate(stages):
        if int(stage["start"]) <= episode <= int(stage["end"]):
            return idx, stage
    return len(stages) - 1, stages[-1]


def _training_score(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["priority_completion_rate"]),
        float(row["completion_rate"]),
        -float(row["collision_count"]),
        -float(row["timeout_count"]),
        float(row["episode_reward"]),
        -float(row["steps"]),
    )


def _evaluate_selection_score(
    model: ActorCritic,
    env_config: WarehouseConfig,
    device: torch.device,
    episodes: int,
) -> tuple[float, ...]:
    rows = []
    eval_env = WarehouseEnv(env_config)
    for _ in progress(range(episodes), total=episodes, desc="select checkpoint", unit="episode", leave=False):
        rollout = _evaluate_greedy_episode(eval_env, model, device)
        rows.append(rollout)
    return (
        _mean(rows, "priority_completion_rate"),
        _mean(rows, "completion_rate"),
        -_mean(rows, "collision_count"),
        -_mean(rows, "timeout_count"),
        _mean(rows, "episode_reward"),
        -_mean(rows, "steps"),
    )


def _evaluate_greedy_episode(env: WarehouseEnv, model: ActorCritic, device: torch.device) -> dict[str, float]:
    observations = env.reset()
    done = False
    total_reward = 0.0
    info: dict[str, Any] = {}
    while not done:
        actions = model.greedy_actions(np.asarray(observations, dtype=np.float32), device)
        actions = env.shield_actions(actions)
        observations, rewards, done, info = env.step(actions)
        total_reward += float(sum(rewards))
    completed = float(info.get("completed_tasks", 0))
    total_tasks = float(len(env.config.tasks))
    completed_priority = float(info.get("completed_priority", 0.0))
    total_priority = float(info.get("total_priority", 0.0))
    return {
        "episode_reward": total_reward,
        "completion_rate": completed / max(total_tasks, 1.0),
        "priority_completion_rate": completed_priority / max(total_priority, 1.0),
        "collision_count": float(info.get("collision_count", 0)),
        "timeout_count": float(info.get("timeout_count", 0)),
        "steps": float(env.step_count),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / max(len(rows), 1)


def _collect_episode(env: WarehouseEnv, model: ActorCritic, device: torch.device) -> dict:
    observations = env.reset()
    done = False
    total_reward = 0.0
    rollout = {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "values": [],
    }
    info = {}

    while not done:
        obs_array = np.asarray(observations, dtype=np.float32)
        actions, log_probs, values = model.act(obs_array, device)
        executed_actions = env.shield_actions(actions.tolist())
        if executed_actions != actions.tolist():
            actions = np.asarray(executed_actions, dtype=np.int64)
            log_probs = model.log_probs_for_actions(obs_array, actions, device)
        next_observations, rewards, done, info = env.step(executed_actions)
        rollout["observations"].append(obs_array)
        rollout["actions"].append(actions)
        rollout["log_probs"].append(log_probs)
        rollout["rewards"].append(np.asarray(rewards, dtype=np.float32))
        rollout["values"].append(values)
        total_reward += float(sum(rewards))
        observations = next_observations

    rollout["episode_reward"] = total_reward
    rollout["info"] = info
    rollout["steps"] = env.step_count
    return rollout


def _behavior_clone_pretrain(
    model: ActorCritic,
    env_config: WarehouseConfig,
    device: torch.device,
    episodes: int,
    epochs: int,
    learning_rate: float,
) -> None:
    env = WarehouseEnv(env_config)
    policy = GreedyTaskPolicy()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    for _ in progress(range(episodes), total=episodes, desc="collect bc data", unit="episode", leave=False):
        episode_obs = env.reset()
        done = False
        while not done:
            obs_array = np.asarray(episode_obs, dtype=np.float32)
            greedy_actions = policy.act(env)
            greedy_actions = env.shield_actions(greedy_actions)
            observations.extend(obs_array)
            actions.extend(greedy_actions)
            episode_obs, _, done, _ = env.step(greedy_actions)

    if not observations:
        return

    obs_tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
    batch_size = min(512, len(actions))
    progress_write(f"behavior cloning samples={len(actions)} epochs={epochs}")
    for epoch in progress(range(1, epochs + 1), total=epochs, desc="behavior cloning", unit="epoch", leave=False):
        permutation = torch.randperm(len(actions), device=device)
        total_loss = 0.0
        for start in range(0, len(actions), batch_size):
            indexes = permutation[start : start + batch_size]
            batch_obs = obs_tensor[indexes]
            batch_actions = action_tensor[indexes]
            logits, _ = model(batch_obs)
            loss = torch.nn.functional.cross_entropy(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indexes)
        if epoch == 1 or epoch == epochs:
            progress_write(f"behavior cloning epoch={epoch}/{epochs} loss={total_loss / len(actions):.4f}")


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _plot_training(history: list[dict], path: Path) -> None:
    episodes = [row["episode"] for row in history]
    rewards = [row["episode_reward"] for row in history]
    completion = [row["completion_rate"] for row in history]
    collisions = [row["collision_count"] for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(episodes, rewards, color="#1f77b4")
    axes[0].set_ylabel("reward")
    axes[1].plot(episodes, completion, color="#2a9d8f")
    axes[1].set_ylabel("completion")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].plot(episodes, collisions, color="#d62728")
    axes[2].set_ylabel("collisions")
    axes[2].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
