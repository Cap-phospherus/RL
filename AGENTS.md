请默认使用中文回答。

## 项目定位

本项目是“智能仓储多机器人拣货调度”的强化学习应用实践。目标不是复刻路径规划示例，而是构建一个可视化、可训练、可评估的多智能体仓储调度系统：

- 多个机器人在二维仓储网格中执行拣货与送货任务。
- 机器人需要避开货架、墙体和其他机器人。
- 任务包含货物位置、打包台位置、截止时间和奖励权重。
- 机器人受电量、移动成本、碰撞风险、等待时间、任务超时等约束影响。
- 最终需要能通过动画或视频直观看到 agent 从混乱随机行为逐步学会完成订单。

## 环境约定

- 默认 Conda 环境：`rl-traffic`
- Python 解释器：`D:\Anaconda\envs\rl-traffic\python.exe`
- 不另建新环境，除非用户明确要求。
- 优先使用当前环境中已有依赖；安装新依赖前必须说明用途、影响和风险。

## 目录约定

- `src/envs/`：仓储网格环境、状态转移、奖励函数。
- `src/agents/`：随机策略、启发式策略、IPPO/PPO 训练代码。
- `src/evaluation/`：任务完成率、平均步数、能耗、碰撞次数、安全性等指标。
- `src/visualization/`：路径图、训练曲线、动画和视频导出。
- `scripts/`：训练、评估、烟测、录制演示视频等入口脚本。
- `configs/`：实验配置。
- `configs/default.yaml` 只作为正式实验组合入口，具体参数在 `configs/full/`。
- `configs/smoke.yaml` 只作为冒烟测试组合入口，具体参数在 `configs/smoke/`。
- `outputs/logs/`：CSV 日志和指标。
- `outputs/figures/`：路径图、指标图、环境快照。
- `outputs/videos/`：运行视频或动画。
- `outputs/models/`：训练后的模型权重。
- `outputs/smoke/`：所有冒烟测试和 `configs/smoke.yaml` 产生的临时输出，不能混入完整实验目录。
- `docs/`：课程报告素材、算法说明、伪代码、流程图。

## 开发规则

- 默认先做只读探索，再修改。
- 修改代码保持最小、清晰、可验证。
- 不要把尚未实现的能力写成已完成能力。
- 如果改动了观测空间、动作空间、奖励函数或模型输入维度，必须同步更新 README、配置和验证脚本。
- 修改正式实验参数时只改 `configs/full/`；修改冒烟测试参数时只改 `configs/smoke/`；不要把长参数重新塞回组合入口文件。
- 训练脚本应支持 smoke 配置，用于快速验证环境、维度和输出路径；smoke 输出必须写到 `outputs/smoke/`。
- 可视化是本项目的核心交付之一；新增算法时要同时考虑如何录制训练前后对比视频。

## 奖励函数原则

奖励函数应体现实际仓储调度目标：

- 成功取货、成功送达给正奖励。
- 移动步数、等待、绕行、电量消耗给成本。
- 碰撞、任务超时、电量耗尽给强惩罚。
- 多机器人挤在同一区域、重复抢同一任务应给惩罚。
- 可以加入安全距离、拥堵区域、任务优先级等扩展项，但必须能在实验指标中解释。

## 常用命令

```powershell
D:\Anaconda\envs\rl-traffic\python.exe scripts\smoke_test.py
D:\Anaconda\envs\rl-traffic\python.exe scripts\eval_fixed.py --config configs\smoke.yaml --device cpu
D:\Anaconda\envs\rl-traffic\python.exe scripts\train_ippo.py --config configs\smoke.yaml --device cpu
D:\Anaconda\envs\rl-traffic\python.exe scripts\diagnose_ippo.py --config configs\smoke.yaml --device cpu
D:\Anaconda\envs\rl-traffic\python.exe scripts\eval_gen.py --config configs\smoke.yaml --device cpu
D:\Anaconda\envs\rl-traffic\python.exe scripts\record_demo.py --config configs\smoke.yaml --policy greedy
D:\Anaconda\envs\rl-traffic\python.exe -m compileall src scripts
```

后续训练入口预期为：

```powershell
D:\Anaconda\envs\rl-traffic\python.exe scripts\train_ippo.py --config configs\default.yaml
D:\Anaconda\envs\rl-traffic\python.exe scripts\eval_fixed.py --config configs\default.yaml
D:\Anaconda\envs\rl-traffic\python.exe scripts\eval_gen.py --config configs\default.yaml
D:\Anaconda\envs\rl-traffic\python.exe scripts\diagnose_ippo.py --config configs\default.yaml
D:\Anaconda\envs\rl-traffic\python.exe scripts\record_demo.py --config configs\default.yaml --policy ippo
```

## 完成说明要求

每次完成任务后，简要说明：

- 做了什么。
- 改在哪里。
- 做了哪些验证。
- 是否还有未完成或需要注意的事项。
