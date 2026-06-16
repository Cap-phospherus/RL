from __future__ import annotations

from collections import deque

from envs import Action, WarehouseEnv


class GreedyTaskPolicy:
    """Shortest-path heuristic baseline with simple conflict dampening."""

    def act(self, env: WarehouseEnv) -> list[int]:
        actions = []
        reserved: set[tuple[int, int]] = set()
        for robot in env.robots:
            task = env.config.tasks[robot.assigned_task] if robot.assigned_task is not None else None
            if robot.carrying_task is None and env._is_low_battery(robot):
                actions.append(int(Action.GO_CHARGE))
                next_position = env._move_for_action(len(actions) - 1, Action.GO_CHARGE)
                reserved.add(next_position)
                continue

            if task is None:
                actions.append(int(Action.SELECT_NEAREST_TASK))
                next_position = env._move_for_action(len(actions) - 1, Action.SELECT_NEAREST_TASK)
                reserved.add(next_position)
                continue

            target = task.dropoff if robot.carrying_task is not None else task.pickup
            action = self._next_action(env, robot.position, target, reserved)
            next_position = env._move(robot.position, Action(action))
            reserved.add(next_position)
            actions.append(action)
        return actions

    def _next_action(
        self,
        env: WarehouseEnv,
        start: tuple[int, int],
        target: tuple[int, int],
        reserved: set[tuple[int, int]],
    ) -> int:
        if start == target:
            return int(Action.STAY)

        queue = deque([start])
        came_from: dict[tuple[int, int], tuple[tuple[int, int], Action] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for action in [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]:
                candidate = env._move(current, action)
                if candidate == current or candidate in came_from:
                    continue
                if candidate in reserved:
                    continue
                came_from[candidate] = (current, action)
                queue.append(candidate)

        if target not in came_from:
            return int(Action.STAY)

        current = target
        previous = came_from[current]
        first_action = Action.STAY
        while previous is not None:
            parent, action = previous
            first_action = action
            current = parent
            previous = came_from[current]
        return int(first_action)
