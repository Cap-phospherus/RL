from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from envs import Task, WarehouseConfig


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return _load_config_file(config_path, stack=[])


def _load_config_file(path: Path, stack: list[Path]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in [*stack, resolved_path])
        raise ValueError(f"Circular config include detected: {chain}")

    with resolved_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {resolved_path}")

    merged: dict[str, Any] = {}
    includes = raw.get("includes", [])
    if includes is None:
        includes = []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    for include in includes:
        include_path = (resolved_path.parent / str(include)).resolve()
        merged = _deep_merge(merged, _load_config_file(include_path, [*stack, resolved_path]))

    local = {key: value for key, value in raw.items() if key != "includes"}
    return _deep_merge(merged, local)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_warehouse_config(path: str | Path) -> WarehouseConfig:
    raw = load_config(path).get("env", {})
    defaults = WarehouseConfig()
    tasks = raw.get("tasks")
    parsed_tasks = defaults.tasks
    if tasks is not None:
        parsed_tasks = [
            Task(
                pickup=tuple(item["pickup"]),
                dropoff=tuple(item["dropoff"]),
                deadline=int(item["deadline"]),
                reward=float(item.get("reward", 20.0)),
                priority=float(item.get("priority", 1.0)),
                release_step=int(item.get("release_step", 0)),
            )
            for item in tasks
        ]

    return WarehouseConfig(
        width=int(raw.get("width", defaults.width)),
        height=int(raw.get("height", defaults.height)),
        num_robots=int(raw.get("num_robots", defaults.num_robots)),
        max_steps=int(raw.get("max_steps", defaults.max_steps)),
        initial_battery=float(raw.get("initial_battery", defaults.initial_battery)),
        seed=int(raw.get("seed", defaults.seed)),
        deterministic_resets=bool(raw.get("deterministic_resets", defaults.deterministic_resets)),
        randomize_starts=bool(raw.get("randomize_starts", defaults.randomize_starts)),
        automatic_task_assignment=bool(raw.get("automatic_task_assignment", defaults.automatic_task_assignment)),
        recharge_rate=float(raw.get("recharge_rate", defaults.recharge_rate)),
        low_battery_threshold=float(raw.get("low_battery_threshold", defaults.low_battery_threshold)),
        congestion_penalty=float(raw.get("congestion_penalty", defaults.congestion_penalty)),
        battery_depletion_penalty=float(
            raw.get("battery_depletion_penalty", defaults.battery_depletion_penalty)
        ),
        shelves=_positions(raw.get("shelves", defaults.shelves)),
        obstacles=_positions(raw.get("obstacles", defaults.obstacles)),
        packing_stations=_positions(raw.get("packing_stations", defaults.packing_stations)),
        charging_stations=_positions(raw.get("charging_stations", defaults.charging_stations)),
        robot_starts=_optional_positions(raw.get("robot_starts")),
        start_scenarios=_optional_position_scenarios(raw.get("start_scenarios")),
        tasks=parsed_tasks,
    )


def _positions(values: list[list[int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [tuple(value) for value in values]


def _optional_positions(values: list[list[int]] | None) -> list[tuple[int, int]] | None:
    if values is None:
        return None
    return _positions(values)


def _optional_position_scenarios(values: list[list[list[int]]] | None) -> list[list[tuple[int, int]]] | None:
    if values is None:
        return None
    return [_positions(scenario) for scenario in values]
