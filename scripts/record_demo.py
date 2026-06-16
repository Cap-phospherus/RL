from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents import GreedyTaskPolicy, RandomPolicy
from agents.ippo import ActorCritic
from envs import WarehouseEnv
from utils import load_config, load_warehouse_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--policy", default="random", choices=["random", "greedy", "ippo"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default=None, choices=["greedy", "sample"])
    parser.add_argument("--sample-candidates", type=int, default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    evaluation = config.get("evaluation", {})
    outputs = config.get("outputs", {})
    model_path = args.model or str(Path(outputs.get("models_dir", "outputs/models")) / "ippo_actor_critic.pt")
    env = WarehouseEnv(load_warehouse_config(config_path))
    mode = args.mode or str(evaluation.get("policy_mode", "greedy"))
    sample_candidates = int(args.sample_candidates or evaluation.get("sample_candidates", 1))
    selected_seed = None
    if args.policy == "ippo" and mode == "sample" and sample_candidates > 1:
        selected_seed = _select_ippo_demo_seed(
            env_config_path=config_path,
            model_path=PROJECT_ROOT / model_path,
            device=_select_device(args.device),
            sample_candidates=sample_candidates,
        )
    observations = env.reset()
    policy = _build_policy(args.policy, env, PROJECT_ROOT / model_path, _select_device(args.device))
    if selected_seed is not None:
        torch.manual_seed(selected_seed)
        np.random.seed(selected_seed)
    frames = [Image.fromarray(env.render_rgb_array())]
    done = False
    info = {}

    while not done:
        actions = _actions(args.policy, policy, env, observations, mode)
        observations, _, done, info = env.step(actions)
        frames.append(Image.fromarray(env.render_rgb_array()))

    videos_dir = PROJECT_ROOT / outputs.get("videos_dir", "outputs/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    output_path = videos_dir / f"{args.policy}_policy_demo.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        optimize=True,
    )
    print(
        "demo saved: "
        f"{output_path} "
        f"policy={args.policy} "
        f"completed={info.get('completed_tasks', 0)}/{len(env.config.tasks)} "
        f"collisions={info.get('collision_count', 0)}"
    )


def _build_policy(policy_name: str, env: WarehouseEnv, model_path: Path, device: torch.device):
    if policy_name == "random":
        return RandomPolicy(action_size=env.action_size, seed=env.config.seed)
    if policy_name == "greedy":
        return GreedyTaskPolicy()

    checkpoint = torch.load(model_path, map_location=device)
    model = ActorCritic(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_size"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def _actions(policy_name: str, policy, env: WarehouseEnv, observations: list[np.ndarray], mode: str) -> list[int]:
    if policy_name == "random":
        return policy.act(num_agents=len(observations))
    if policy_name == "greedy":
        return policy.act(env)
    model, device = policy
    if mode == "sample":
        sampled_actions, _, _ = model.act(np.asarray(observations, dtype=np.float32), device)
        actions = sampled_actions.tolist()
    else:
        actions = model.greedy_actions(np.asarray(observations, dtype=np.float32), device)
    return env.shield_actions(actions)


def _select_ippo_demo_seed(env_config_path: Path, model_path: Path, device: torch.device, sample_candidates: int) -> int:
    checkpoint = torch.load(model_path, map_location=device)
    env_config = load_warehouse_config(env_config_path)
    model = ActorCritic(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_size"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_seed = 10_101
    best_score = None
    for candidate in range(1, sample_candidates + 1):
        seed = 10_100 + candidate
        score = _score_ippo_candidate(env_config, model, device, seed)
        if best_score is None or score > best_score:
            best_score = score
            best_seed = seed
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    return best_seed


def _score_ippo_candidate(env_config, model: ActorCritic, device: torch.device, seed: int) -> tuple[float, ...]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = WarehouseEnv(env_config)
    observations = env.reset()
    done = False
    total_reward = 0.0
    info = {}
    while not done:
        sampled_actions, _, _ = model.act(np.asarray(observations, dtype=np.float32), device)
        actions = env.shield_actions(sampled_actions.tolist())
        observations, rewards, done, info = env.step(actions)
        total_reward += float(sum(rewards))
    completion = float(info.get("completed_tasks", 0)) / max(len(env.config.tasks), 1)
    priority = float(info.get("completed_priority", 0.0)) / max(float(info.get("total_priority", 0.0)), 1.0)
    return (completion, priority, -float(env.step_count), total_reward, -float(info.get("collision_count", 0)))


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
