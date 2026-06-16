from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


Position = tuple[int, int]


class Action(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SELECT_NEAREST_TASK = 5
    SELECT_PRIORITY_TASK = 6
    GO_CHARGE = 7


@dataclass(frozen=True)
class Task:
    pickup: Position
    dropoff: Position
    deadline: int
    reward: float = 20.0
    priority: float = 1.0
    release_step: int = 0


@dataclass
class RobotState:
    position: Position
    battery: float
    carrying_task: int | None = None
    assigned_task: int | None = None
    completed_tasks: int = 0


@dataclass
class WarehouseConfig:
    width: int = 8
    height: int = 6
    num_robots: int = 2
    max_steps: int = 80
    initial_battery: float = 100.0
    seed: int = 7
    deterministic_resets: bool = True
    randomize_starts: bool = False
    automatic_task_assignment: bool = True
    recharge_rate: float = 8.0
    low_battery_threshold: float = 0.25
    congestion_penalty: float = 0.15
    battery_depletion_penalty: float = 8.0
    shelves: list[Position] = field(
        default_factory=lambda: [(2, 1), (2, 2), (2, 3), (5, 1), (5, 2), (5, 3)]
    )
    obstacles: list[Position] = field(default_factory=list)
    packing_stations: list[Position] = field(default_factory=lambda: [(7, 0)])
    charging_stations: list[Position] = field(default_factory=lambda: [(0, 5)])
    robot_starts: list[Position] | None = None
    start_scenarios: list[list[Position]] | None = None
    tasks: list[Task] = field(
        default_factory=lambda: [
            Task(pickup=(2, 1), dropoff=(7, 0), deadline=50, reward=20.0),
            Task(pickup=(5, 3), dropoff=(7, 0), deadline=70, reward=25.0),
        ]
    )


class WarehouseEnv:
    """Small synchronous multi-robot warehouse environment.

    The implementation is intentionally lightweight so the project has a
    runnable baseline before PPO/IPPO training is added.
    """

    action_size = len(Action)

    def __init__(self, config: WarehouseConfig | None = None) -> None:
        self.config = config or WarehouseConfig()
        self.rng = random.Random(self.config.seed)
        self.step_count = 0
        self.robots: list[RobotState] = []
        self.completed_task_ids: set[int] = set()
        self.timed_out_task_ids: set[int] = set()
        self.collision_count = 0
        self.timeout_count = 0
        self.reset_count = 0
        self.robot_paths: list[list[Position]] = []

    @property
    def observation_dim(self) -> int:
        return 21

    def reset(self) -> list[np.ndarray]:
        if self.config.deterministic_resets:
            self.rng.seed(self.config.seed)
        self.step_count = 0
        self.completed_task_ids = set()
        self.timed_out_task_ids = set()
        self.collision_count = 0
        self.timeout_count = 0
        starts = self._select_robot_starts()
        self.reset_count += 1
        self.robots = [
            RobotState(position=starts[i], battery=self.config.initial_battery)
            for i in range(self.config.num_robots)
        ]
        self.robot_paths = [[robot.position] for robot in self.robots]
        if self.config.automatic_task_assignment:
            self._assign_tasks()
        return self.observe()

    def observe(self) -> list[np.ndarray]:
        observations = []
        occupied = {robot.position for robot in self.robots}
        for robot in self.robots:
            task = self._active_task(robot)
            target = robot.position
            carrying = 0.0
            deadline_left = 0.0
            priority = 0.0
            if task is not None:
                target = task.dropoff if robot.carrying_task is not None else task.pickup
                carrying = float(robot.carrying_task is not None)
                deadline_left = max(task.deadline - self.step_count, 0) / max(task.deadline, 1)
                priority = task.priority / max(self._max_task_priority(), 1.0)
            elif self.config.charging_stations and self._is_low_battery(robot):
                target = self._nearest_charger(robot.position)

            x, y = robot.position
            tx, ty = target
            nearby_blocked = self._nearby_blocked_count(robot.position, occupied)
            charger = self._nearest_charger(robot.position)
            cx, cy = charger
            max_distance = max(self.config.width + self.config.height - 2, 1)
            target_distance = self._manhattan(robot.position, target) / max_distance
            charger_distance = self._manhattan(robot.position, charger) / max_distance
            available_pressure = len(self._available_task_ids()) / max(len(self.config.tasks), 1)
            candidate_task = self._best_available_task(robot.position, mode="priority")
            candidate_dx = 0.0
            candidate_dy = 0.0
            candidate_priority = 0.0
            candidate_deadline = 0.0
            if candidate_task is not None:
                candidate_dx = (candidate_task.pickup[0] - x) / max(self.config.width - 1, 1)
                candidate_dy = (candidate_task.pickup[1] - y) / max(self.config.height - 1, 1)
                candidate_priority = candidate_task.priority / max(self._max_task_priority(), 1.0)
                candidate_deadline = max(candidate_task.deadline - self.step_count, 0) / max(candidate_task.deadline, 1)
            low_battery = float(
                robot.battery
                <= self.config.initial_battery * self.config.low_battery_threshold
            )
            obs = np.array(
                [
                    x / max(self.config.width - 1, 1),
                    y / max(self.config.height - 1, 1),
                    (tx - x) / max(self.config.width - 1, 1),
                    (ty - y) / max(self.config.height - 1, 1),
                    robot.battery / max(self.config.initial_battery, 1.0),
                    carrying,
                    deadline_left,
                    nearby_blocked / 4.0,
                    len(self.completed_task_ids) / max(len(self.config.tasks), 1),
                    self.step_count / max(self.config.max_steps, 1),
                    target_distance,
                    (cx - x) / max(self.config.width - 1, 1),
                    (cy - y) / max(self.config.height - 1, 1),
                    charger_distance,
                    priority,
                    available_pressure,
                    low_battery,
                    candidate_dx,
                    candidate_dy,
                    candidate_priority,
                    candidate_deadline,
                ],
                dtype=np.float32,
            )
            observations.append(obs)
        return observations

    def step(self, actions: list[int]) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        if len(actions) != len(self.robots):
            raise ValueError(f"Expected {len(self.robots)} actions, got {len(actions)}")

        self.step_count += 1
        rewards = [-0.04 for _ in self.robots]
        self._handle_high_level_actions(actions, rewards)
        previous_targets = [self._current_target(robot) for robot in self.robots]
        previous_distances = [
            self._manhattan(robot.position, target) if target is not None else None
            for robot, target in zip(self.robots, previous_targets)
        ]
        previous_charger_distances = [
            self._manhattan(robot.position, self._nearest_charger(robot.position))
            for robot in self.robots
        ]
        proposed = [
            self._move_for_action(idx, Action(action))
            if robot.battery > 0.0
            else robot.position
            for idx, (robot, action) in enumerate(zip(self.robots, actions))
        ]
        proposed, collision_flags = self._resolve_collisions(proposed)

        for idx, robot in enumerate(self.robots):
            action = Action(actions[idx])
            if robot.battery <= 0.0 and action != Action.STAY:
                rewards[idx] -= self.config.battery_depletion_penalty
            elif action != Action.STAY:
                rewards[idx] -= 0.03
                robot.battery -= 1.0
            else:
                rewards[idx] -= 0.02
                robot.battery -= 0.2
                if self._active_task(robot) is not None and robot.position not in set(self.config.charging_stations):
                    rewards[idx] -= 0.15

            if collision_flags[idx]:
                rewards[idx] -= 8.0
                robot.battery -= 2.0
                self.collision_count += 1
            robot.position = proposed[idx]
            if previous_targets[idx] is not None and previous_distances[idx] is not None:
                current_distance = self._manhattan(robot.position, previous_targets[idx])
                task = self._active_task(robot)
                priority = task.priority if task is not None else 1.0
                rewards[idx] += 0.25 * priority * (previous_distances[idx] - current_distance)
            if self._is_low_battery(robot):
                current_charger_distance = self._manhattan(robot.position, self._nearest_charger(robot.position))
                rewards[idx] += 0.35 * (previous_charger_distances[idx] - current_charger_distance)

            nearby_robots = self._nearby_robot_count(idx)
            rewards[idx] -= self.config.congestion_penalty * nearby_robots
            if self._is_low_battery(robot) and robot.position not in set(self.config.charging_stations):
                rewards[idx] -= 0.25
            if robot.position in set(self.config.charging_stations):
                was_low = self._is_low_battery(robot)
                robot.battery = min(self.config.initial_battery, robot.battery + self.config.recharge_rate)
                if was_low:
                    rewards[idx] += 0.8
            robot.battery = max(0.0, robot.battery)
        for idx, robot in enumerate(self.robots):
            if idx < len(self.robot_paths):
                self.robot_paths[idx].append(robot.position)

        self._handle_pickups_and_dropoffs(rewards)
        self._handle_timeouts(rewards)
        if self.config.automatic_task_assignment:
            self._assign_tasks()

        done = self.step_count >= self.config.max_steps or len(self.completed_task_ids) == len(self.config.tasks)
        info = {
            "completed_tasks": len(self.completed_task_ids),
            "completed_priority": self._completed_priority(),
            "total_priority": sum(task.priority for task in self.config.tasks),
            "collision_count": self.collision_count,
            "timeout_count": self.timeout_count,
            "total_battery": sum(robot.battery for robot in self.robots),
        }
        return self.observe(), rewards, done, info

    def shield_actions(self, actions: list[int]) -> list[int]:
        """Convert immediately conflicting actions to STAY without mutating state."""
        current = [robot.position for robot in self.robots]
        proposed = [
            self._project_position_for_action(idx, Action(action))
            if self.robots[idx].battery > 0.0
            else self.robots[idx].position
            for idx, action in enumerate(actions)
        ]
        uncertain = [
            Action(action) in {Action.SELECT_NEAREST_TASK, Action.SELECT_PRIORITY_TASK}
            and self.robots[idx].assigned_task is None
            and self.robots[idx].carrying_task is None
            for idx, action in enumerate(actions)
        ]
        shielded = actions[:]
        reserved: dict[Position, int] = {}
        for idx, pos in enumerate(proposed):
            if uncertain[idx]:
                continue
            if pos in reserved:
                shielded[idx] = int(Action.STAY)
                continue
            reserved[pos] = idx
        for i in range(len(proposed)):
            for j in range(i + 1, len(proposed)):
                if uncertain[i] or uncertain[j]:
                    continue
                if proposed[i] == current[j] and proposed[j] == current[i]:
                    shielded[j] = int(Action.STAY)
        return shielded

    def save_snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def render_rgb_array(self) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw(ax)
        fig.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image

    def _draw(self, ax: plt.Axes) -> None:
        robot_colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
        ax.set_xlim(-0.5, self.config.width - 0.5)
        ax.set_ylim(self.config.height - 0.5, -0.5)
        ax.set_xticks(range(self.config.width))
        ax.set_yticks(range(self.config.height))
        ax.set_facecolor("#f8fafc")
        ax.grid(True, color="#cbd5e1", linewidth=0.7)
        ax.tick_params(labelsize=7, colors="#475569")

        for x, y in self.config.shelves:
            ax.add_patch(
                plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color="#8b6f47", alpha=0.82, zorder=1)
            )
            ax.text(x, y, "S", ha="center", va="center", color="white", fontsize=7, weight="bold", zorder=2)
        for x, y in self.config.packing_stations:
            ax.add_patch(
                plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color="#0f766e", alpha=0.9, zorder=1)
            )
            ax.text(x, y, "P", ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=2)
        for x, y in self.config.charging_stations:
            ax.add_patch(
                plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color="#f59e0b", alpha=0.9, zorder=1)
            )
            ax.text(x, y, "C", ha="center", va="center", color="#1f2937", fontsize=8, weight="bold", zorder=2)

        released_task_ids = [
            idx
            for idx, task in enumerate(self.config.tasks)
            if idx not in self.completed_task_ids and task.release_step <= self.step_count
        ]
        for task_id in released_task_ids:
            task = self.config.tasks[task_id]
            px, py = task.pickup
            dx, dy = task.dropoff
            priority_alpha = min(1.0, 0.35 + task.priority / max(self._max_task_priority(), 1.0) * 0.55)
            ax.scatter([px], [py], s=145, marker="D", color="#facc15", edgecolor="#713f12", alpha=priority_alpha, zorder=3)
            ax.text(px, py - 0.02, f"T{task_id}", ha="center", va="center", fontsize=6, color="#422006", zorder=4)
            ax.scatter([dx], [dy], s=130, marker="*", color="#14b8a6", edgecolor="#0f766e", alpha=0.78, zorder=3)

        for idx, path in enumerate(self.robot_paths):
            if len(path) < 2:
                continue
            color = robot_colors[idx % len(robot_colors)]
            trail = path[-50:]
            xs = [position[0] for position in trail]
            ys = [position[1] for position in trail]
            ax.plot(xs, ys, color=color, alpha=0.32, linewidth=2.1, zorder=2)

        for idx, robot in enumerate(self.robots):
            task = self._active_task(robot)
            if task is None:
                continue
            target = task.dropoff if robot.carrying_task is not None else task.pickup
            color = robot_colors[idx % len(robot_colors)]
            ax.plot(
                [robot.position[0], target[0]],
                [robot.position[1], target[1]],
                color=color,
                alpha=0.18,
                linewidth=1.6,
                linestyle="--",
                zorder=2,
            )

        for idx, robot in enumerate(self.robots):
            x, y = robot.position
            color = robot_colors[idx % len(robot_colors)]
            battery_ratio = max(0.0, min(1.0, robot.battery / max(self.config.initial_battery, 1.0)))
            depleted = battery_ratio <= 1e-6
            edge_color = "#111827" if depleted else "#0f172a"
            ring_color = "#7f1d1d" if depleted else "#22c55e" if battery_ratio > 0.35 else "#ef4444"
            ax.scatter([x], [y], s=390, marker="o", color=color, edgecolor=edge_color, linewidth=1.3, zorder=5)
            ax.scatter([x], [y], s=520, marker="o", facecolor="none", edgecolor=ring_color, linewidth=2.0, alpha=0.85, zorder=4)
            ax.text(x, y - 0.05, f"R{idx}", ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=6)
            ax.text(
                x,
                y + 0.34,
                "OFF" if depleted else f"{int(battery_ratio * 100)}%",
                ha="center",
                va="center",
                color="#7f1d1d" if depleted else "#0f172a",
                fontsize=6,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                zorder=6,
            )
            if depleted:
                ax.text(x, y + 0.08, "x", ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=7)

        ax.set_title(
            "Warehouse dispatch | "
            f"step {self.step_count}/{self.config.max_steps} | "
            f"completed {len(self.completed_task_ids)}/{len(self.config.tasks)} | "
            f"collisions {self.collision_count} | timeouts {self.timeout_count}",
            fontsize=11,
            color="#0f172a",
            pad=10,
        )
        ax.set_aspect("equal")

    def _select_robot_starts(self) -> list[Position]:
        if self.config.start_scenarios:
            scenario = self.config.start_scenarios[self.reset_count % len(self.config.start_scenarios)]
            return scenario[: self.config.num_robots]
        if self.config.robot_starts is not None:
            return self.config.robot_starts[: self.config.num_robots]
        if not self.config.randomize_starts:
            return self._default_robot_starts()
        free_cells = [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if self._is_free((x, y))
            and (x, y) not in set(self.config.shelves)
            and (x, y) not in set(self.config.packing_stations)
        ]
        self.rng.shuffle(free_cells)
        return free_cells[: self.config.num_robots]

    def _default_robot_starts(self) -> list[Position]:
        candidates = [(0, 0), (0, self.config.height - 1), (1, 0), (1, self.config.height - 1)]
        return candidates[: self.config.num_robots]

    def _move(self, position: Position, action: Action) -> Position:
        x, y = position
        deltas = {
            Action.STAY: (0, 0),
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
            Action.SELECT_NEAREST_TASK: (0, 0),
            Action.SELECT_PRIORITY_TASK: (0, 0),
            Action.GO_CHARGE: (0, 0),
        }
        dx, dy = deltas[action]
        next_pos = (x + dx, y + dy)
        if not self._is_free(next_pos):
            return position
        return next_pos

    def _move_for_action(self, robot_idx: int, action: Action) -> Position:
        robot = self.robots[robot_idx]
        if action in {Action.SELECT_NEAREST_TASK, Action.SELECT_PRIORITY_TASK, Action.GO_CHARGE}:
            target = self._current_target(robot)
            if action == Action.GO_CHARGE or target is None:
                target = self._nearest_charger(robot.position)
            next_action = self._next_step_action(robot.position, target)
            return self._move(robot.position, next_action)
        return self._move(robot.position, action)

    def _project_position_for_action(self, robot_idx: int, action: Action) -> Position:
        robot = self.robots[robot_idx]
        if action == Action.SELECT_NEAREST_TASK and robot.carrying_task is None:
            task = self._best_available_task(robot.position, mode="nearest")
            target = task.pickup if task is not None else self._nearest_charger(robot.position)
            return self._move(robot.position, self._next_step_action(robot.position, target))
        if action == Action.SELECT_PRIORITY_TASK and robot.carrying_task is None:
            task = self._best_available_task(robot.position, mode="priority")
            target = task.pickup if task is not None else self._nearest_charger(robot.position)
            return self._move(robot.position, self._next_step_action(robot.position, target))
        if action == Action.GO_CHARGE:
            target = self._nearest_charger(robot.position)
            return self._move(robot.position, self._next_step_action(robot.position, target))
        return self._move_for_action(robot_idx, action)

    def _is_free(self, position: Position) -> bool:
        x, y = position
        if x < 0 or y < 0 or x >= self.config.width or y >= self.config.height:
            return False
        return position not in set(self.config.obstacles)

    def _resolve_collisions(self, proposed: list[Position]) -> tuple[list[Position], list[bool]]:
        resolved = proposed[:]
        collisions = [False] * len(proposed)
        current = [robot.position for robot in self.robots]

        for i, pos in enumerate(proposed):
            if proposed.count(pos) > 1:
                resolved[i] = current[i]
                collisions[i] = True

        for i in range(len(proposed)):
            for j in range(i + 1, len(proposed)):
                if proposed[i] == current[j] and proposed[j] == current[i]:
                    resolved[i] = current[i]
                    resolved[j] = current[j]
                    collisions[i] = True
                    collisions[j] = True
        return resolved, collisions

    def _handle_pickups_and_dropoffs(self, rewards: list[float]) -> None:
        for idx, robot in enumerate(self.robots):
            task = self._active_task(robot)
            if task is None:
                continue
            task_id = robot.assigned_task
            if robot.carrying_task is None and robot.position == task.pickup:
                robot.carrying_task = task_id
                rewards[idx] += 6.0 * task.priority
            elif robot.carrying_task == task_id and robot.position == task.dropoff:
                self.completed_task_ids.add(task_id)
                robot.carrying_task = None
                robot.assigned_task = None
                robot.completed_tasks += 1
                slack = max(task.deadline - self.step_count, 0)
                slack_bonus = 0.03 * task.priority * slack
                rewards[idx] += 2.0 * task.reward * task.priority + slack_bonus

    def _handle_timeouts(self, rewards: list[float]) -> None:
        for idx, robot in enumerate(self.robots):
            task = self._active_task(robot)
            if task is None:
                continue
            if self.step_count > task.deadline and robot.assigned_task not in self.completed_task_ids:
                rewards[idx] -= 0.1 * task.priority
                if robot.assigned_task not in self.timed_out_task_ids:
                    rewards[idx] -= 6.0 * task.priority
                    self.timed_out_task_ids.add(robot.assigned_task)
                    self.timeout_count += 1

    def _assign_tasks(self) -> None:
        assigned = {robot.assigned_task for robot in self.robots if robot.assigned_task is not None}
        available = [
            idx
            for idx in range(len(self.config.tasks))
            if idx not in self.completed_task_ids
            and idx not in assigned
            and self.config.tasks[idx].release_step <= self.step_count
        ]
        available.sort(key=lambda idx: (-self.config.tasks[idx].priority, self.config.tasks[idx].deadline))
        for robot in self.robots:
            if robot.assigned_task is None and available:
                robot.assigned_task = available.pop(0)

    def _handle_high_level_actions(self, actions: list[int], rewards: list[float]) -> None:
        for idx, (robot, raw_action) in enumerate(zip(self.robots, actions)):
            action = Action(raw_action)
            if action == Action.GO_CHARGE:
                if robot.carrying_task is None:
                    robot.assigned_task = None
                if self._is_low_battery(robot):
                    rewards[idx] += 0.2
                continue
            if action not in {Action.SELECT_NEAREST_TASK, Action.SELECT_PRIORITY_TASK}:
                continue
            if robot.carrying_task is not None or robot.assigned_task is not None:
                continue
            mode = "nearest" if action == Action.SELECT_NEAREST_TASK else "priority"
            task_id = self._best_available_task_id(robot.position, mode=mode)
            if task_id is not None:
                if robot.assigned_task is None:
                    rewards[idx] += 0.2
                robot.assigned_task = task_id

    def _active_task(self, robot: RobotState) -> Task | None:
        if robot.assigned_task is None:
            return None
        if robot.assigned_task in self.completed_task_ids:
            return None
        return self.config.tasks[robot.assigned_task]

    def _current_target(self, robot: RobotState) -> Position | None:
        task = self._active_task(robot)
        if task is None:
            return None
        return task.dropoff if robot.carrying_task is not None else task.pickup

    def _next_step_action(self, start: Position, target: Position | None) -> Action:
        if target is None or start == target:
            return Action.STAY
        candidates = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        best_action = Action.STAY
        best_distance = self._manhattan(start, target)
        for action in candidates:
            candidate = self._move(start, action)
            if candidate == start:
                continue
            distance = self._manhattan(candidate, target)
            if distance < best_distance:
                best_distance = distance
                best_action = action
        return best_action

    def _manhattan(self, left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _nearby_blocked_count(self, position: Position, occupied: set[Position]) -> int:
        x, y = position
        count = 0
        for candidate in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            if not self._is_free(candidate) or candidate in occupied:
                count += 1
        return count

    def _available_task_ids(self) -> list[int]:
        assigned = {robot.assigned_task for robot in self.robots if robot.assigned_task is not None}
        return [
            idx
            for idx, task in enumerate(self.config.tasks)
            if idx not in self.completed_task_ids
            and idx not in assigned
            and task.release_step <= self.step_count
        ]

    def _best_available_task_id(self, position: Position, mode: str) -> int | None:
        available = self._available_task_ids()
        if not available:
            return None
        if mode == "nearest":
            return min(
                available,
                key=lambda idx: (
                    self._manhattan(position, self.config.tasks[idx].pickup),
                    -self.config.tasks[idx].priority,
                    self.config.tasks[idx].deadline,
                ),
            )
        return min(
            available,
            key=lambda idx: (
                -self.config.tasks[idx].priority,
                self.config.tasks[idx].deadline,
                self._manhattan(position, self.config.tasks[idx].pickup),
            ),
        )

    def _best_available_task(self, position: Position, mode: str) -> Task | None:
        task_id = self._best_available_task_id(position, mode=mode)
        if task_id is None:
            return None
        return self.config.tasks[task_id]

    def _nearest_charger(self, position: Position) -> Position:
        if not self.config.charging_stations:
            return position
        return min(self.config.charging_stations, key=lambda station: self._manhattan(position, station))

    def _max_task_priority(self) -> float:
        if not self.config.tasks:
            return 1.0
        return max(task.priority for task in self.config.tasks)

    def _completed_priority(self) -> float:
        return sum(self.config.tasks[idx].priority for idx in self.completed_task_ids)

    def _nearby_robot_count(self, robot_idx: int) -> int:
        position = self.robots[robot_idx].position
        return sum(
            1
            for idx, robot in enumerate(self.robots)
            if idx != robot_idx and self._manhattan(position, robot.position) <= 1
        )

    def _is_low_battery(self, robot: RobotState) -> bool:
        return robot.battery <= self.config.initial_battery * self.config.low_battery_threshold
