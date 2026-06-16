# 报告结构草案

## 1. 项目背景与目标

- 研究对象是二维仓储网格中的多机器人拣货调度问题。
- 系统目标是在有限步数和有限电量下完成动态订单，并降低碰撞、拥堵和无效移动。
- 项目重点不是单机器人路径规划，而是多机器人任务分配、协同避让、电量约束和可视化对比。

## 2. 环境建模与基本假设

- 仓库地图使用 `12 x 8` 网格表示，包含通道、货架服务格、打包台、充电站和可选障碍物。
- 当前正式场景包含 `4` 个机器人、`12` 个动态释放订单、`2` 个打包台、`2` 个充电站和 `5` 组训练固定起点。
- 每个订单包含取货点、送达点、截止时间、基础奖励、优先级和释放时间。
- 机器人每步执行一个离散动作，移动、等待、碰撞和停机都会影响奖励与电量。
- 电量耗尽后机器人原地停机，可视化中显示为 `OFF`。

## 3. 强化学习建模

- 每个机器人被视为一个 agent，使用共享策略参数的 IPPO 进行训练。
- 单机器人观测为 21 维，包含自身位置、电量、携带状态、目标方向、截止时间、附近拥堵、完成进度、充电站方向和候选任务信息。
- 动作空间包含 8 个离散动作：停留、上、下、左、右、选择最近任务、选择最高优先级任务、前往充电站。
- 奖励函数由任务完成奖励、优先级权重、目标进度奖励、充电进度奖励、移动成本、等待成本、碰撞惩罚、拥堵惩罚、超时惩罚和电量耗尽惩罚组成。

## 4. 算法设计

- Random 策略作为无学习下界，用于展示无调度策略的失败行为。
- Greedy 策略作为启发式 baseline，按最短路和简单充电规则直接决策，不需要训练。
- IPPO 使用共享 actor-critic 网络，各机器人基于局部观测独立采样动作。
- 动作 mask 用于过滤明显无效动作，collision shield 用于执行前屏蔽明显同格或互换位置冲突。
- 训练采用 curriculum 与 checkpoint 选择，正式配置当前为 700 回合短训练。
- 正式随机起点泛化评估使用 `20` 组训练未见起点，IPPO sample+shield 每组使用 `8` 个候选 rollout，并按完成率、优先级完成率、步数、奖励和碰撞筛选。

## 5. 工程实现

- `src/envs/` 实现仓储环境、任务流程、奖励函数、碰撞处理、电量和渲染。
- `src/agents/` 实现 random、greedy 和 IPPO 策略。
- `src/evaluation/` 实现 episode 指标统计。
- `src/utils/` 实现 YAML 配置读取和环境配置转换。
- `scripts/smoke_test.py` 提供环境和基础策略冒烟测试。
- `scripts/train_ippo.py` 提供 IPPO 训练入口。
- `scripts/eval_fixed.py` 提供固定训练起点公平对比，输出 `fixed_eval_rows.csv` 和 `fixed_eval_summary.csv`。
- `scripts/eval_gen.py` 提供随机起点泛化评估，输出 `generalization_rows.csv`、`generalization_starts.csv`、`generalization_summary.csv` 和 `generalization_verdict.txt`。
- `scripts/diagnose_ippo.py` 只用于 checkpoint 快速诊断，不作为正式主评估入口。
- `scripts/record_demo.py` 提供 GIF 录制入口。
- `configs/default.yaml` 是正式实验组合入口，具体参数拆分在 `configs/full/`；`configs/smoke.yaml` 是快速验证组合入口，具体参数拆分在 `configs/smoke/`。
- 正式输出写入 `outputs/logs`、`outputs/figures`、`outputs/models` 和 `outputs/videos`；冒烟输出统一隔离到 `outputs/smoke/`。

## 6. 仿真与性能评估

评估指标：

- `completion_rate`：订单完成率。
- `priority_completion_rate`：优先级加权完成率。
- `episode_reward`：平均回合奖励。
- `collision_count`：碰撞次数。
- `timeout_count`：超时任务数。
- `steps`：完成或终止步数。
- `total_battery`：所有机器人剩余电量总和。

正式随机起点泛化评估结果：

| 策略 | 完成率 | 优先级完成率 | 平均奖励 | 平均碰撞 | 平均步数 |
|---|---:|---:|---:|---:|---:|
| IPPO sample+shield | 0.9875 | 0.9867 | 535.17 | 40.9 | 228.6 |
| Greedy+shield | 0.8917 | 0.8777 | -4784.05 | 191.5 | 266.15 |
| Greedy | 0.9167 | 0.9101 | -7661.23 | 335.05 | 304.75 |
| Random | 0.4625 | 0.4399 | -7621.21 | 52.9 | 420.0 |

该评估使用 `20` 组训练未见随机起点，`IPPO sample+shield` 每组采样 `8` 个候选轨迹。完成率 95% 置信区间为 `[0.9741, 1.0000]`，优先级完成率 95% 置信区间为 `[0.9720, 1.0000]`，平均碰撞为 `40.9`，满足当前阈值判定并得到 `PASS`。

固定训练起点公平对比结果：

| 策略 | 完成率 | 优先级完成率 | 平均奖励 | 平均碰撞 | 平均步数 |
|---|---:|---:|---:|---:|---:|
| IPPO sample+shield | 1.0 | 1.0 | 897.40 | 34.2 | 166.0 |
| Greedy+shield | 1.0 | 1.0 | -1362.61 | 117.2 | 171.0 |
| Greedy | 1.0 | 1.0 | -2500.50 | 185.2 | 171.0 |
| IPPO greedy+shield | 0.2667 | 0.2127 | -927.83 | 16.4 | 420.0 |
| Random | 0.4167 | 0.3787 | -7809.67 | 59.2 | 420.0 |

固定 5 组起点参与训练，只用于训练分布内对照，不能作为泛化证明。

## 7. 可视化展示

- `outputs/figures/ippo_training_curves.png` 展示 IPPO 训练过程中的奖励、完成率和碰撞变化。
- `outputs/figures/smoke_final_state.png` 展示冒烟测试结束时的环境状态。
- `outputs/videos/random_policy_demo.gif` 展示随机策略在复杂场景下的低效和停机行为。
- `outputs/videos/greedy_policy_demo.gif` 展示启发式策略虽然能完成任务，但碰撞较多。
- `outputs/videos/ippo_policy_demo.gif` 展示训练后 IPPO 策略在完成率、碰撞控制和电量保留上的优势。
- GIF 中显示货架、打包台、充电站、任务点、机器人编号、电量或 `OFF` 状态、最近轨迹、目标连线和关键指标。

## 8. 结论

- 随机策略无法稳定完成动态订单，说明任务具有足够难度。
- Greedy 在随机起点上完成率较高，但缺少多机器人协同，碰撞和奖励表现较差。
- Greedy+shield 能降低碰撞，但仍不能兼顾完成率、奖励和安全性。
- IPPO sample+shield 在训练未见随机起点上保持高完成率和高优先级完成率，同时显著降低碰撞。
- 当前结果可以支撑“强化学习策略结合安全 shield 和候选采样后，在多机器人仓储调度中能学习更优协同模式”的报告结论。

## 9. 局限与展望

- 当前泛化结论来自有限随机起点的统计验证，不是对所有可能起点的数学证明。
- IPPO greedy+shield 表现较弱，说明确定性 argmax 执行仍不稳定，最终强结果依赖候选采样与 collision shield。
- 后续可引入 centralized critic、分层策略、更多地图规模、更多随机种子和奖励消融实验。
- 可进一步扩展不可通行货架、动态障碍、订单在线到达和更真实的充电调度约束。
