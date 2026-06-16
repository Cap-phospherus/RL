# 智能仓储多机器人拣货调度强化学习实践

本项目实现了一个可训练、可评估、可视化的多机器人仓储调度环境。4 个机器人需要在二维网格仓库中完成动态释放的拣货订单，并在有限步数、电量、充电站和碰撞约束下尽量提高完成率、减少拥堵和提升总奖励。

当前正式环境使用 `configs/default.yaml`：`12 x 8` 地图、4 个机器人、12 个动态订单、双打包台、双充电站、5 组训练固定起点和任务优先级。正式评估以训练未见随机起点为主，固定起点结果仅作为训练分布内对照。

## 当前结论

随机起点泛化评估使用 20 组训练未见起点，IPPO sample 每组采样 8 个候选轨迹：

| 策略 | 完成率 | 优先级完成率 | 平均奖励 | 平均碰撞 | 平均步数 |
|---|---:|---:|---:|---:|---:|
| IPPO sample+shield | 0.9875 | 0.9867 | 535.17 | 40.9 | 228.6 |
| Greedy+shield | 0.8917 | 0.8777 | -4784.05 | 191.5 | 266.15 |
| Greedy | 0.9167 | 0.9101 | -7661.23 | 335.05 | 304.75 |
| Random | 0.4625 | 0.4399 | -7621.21 | 52.9 | 420.0 |

结论：随机起点评估更能检验泛化能力。IPPO sample+shield 在训练未见起点上保持高完成率和高优先级完成率，同时碰撞显著低于 Greedy 和 Greedy+shield。固定 5 组起点由于参与训练，只用于观察训练分布内调度质量，不作为主要泛化结论。

## 核心特性

- 多机器人同步移动、碰撞检测和互换位置冲突检测。
- 动态订单释放、任务优先级、截止时间和加权奖励。
- 电量消耗、低电量充电、耗尽停机和 `OFF` 可视化。
- Random、Greedy 和 IPPO 三类策略对比。
- IPPO 支持 curriculum、动作 mask、collision shield 和候选采样评估。
- GIF 可视化显示货架、打包台、充电站、任务点、机器人编号、电量、轨迹、目标连线和关键指标。

## 项目结构

```text
configs/
  default.yaml              正式实验组合入口
  smoke.yaml                快速验证组合入口
  full/                     正式实验参数：环境、训练、课程、评估、泛化和输出路径
  smoke/                    冒烟测试参数：小环境、短训练、短评估和隔离输出路径
docs/
  optimization_plan.md      IPPO 优化记录和实验分析
  project_outline.md        报告结构草案
scripts/
  smoke_test.py             冒烟测试
  eval_fixed.py             固定训练起点公平对比
  train_ippo.py             IPPO 训练
  diagnose_ippo.py          IPPO checkpoint 快速诊断
  eval_gen.py               随机起点泛化评估
  record_demo.py            策略 GIF 录制
src/
  agents/                   Random、Greedy、IPPO 策略
  envs/                     仓储环境、奖励、碰撞、电量、渲染
  evaluation/               指标统计
  utils/                    配置读取
outputs/
  figures/                  训练曲线和环境截图
  logs/                     CSV 指标日志
  models/                   IPPO checkpoint
  videos/                   策略演示 GIF
```

## 常用命令

冒烟测试：

```powershell
python scripts\smoke_test.py
```

冒烟测试和 `configs/smoke.yaml` 相关输出会写入 `outputs/smoke/`，不会混入正式实验目录。

固定训练起点公平对比：

```powershell
python scripts\eval_fixed.py --config configs\default.yaml
```

训练 IPPO：

```powershell
python scripts\train_ippo.py --config configs\default.yaml
```

诊断 IPPO checkpoint：

```powershell
python scripts\diagnose_ippo.py --config configs\default.yaml
```

随机起点泛化评估：

```powershell
python scripts\eval_gen.py --config configs\default.yaml
```

快速验证版：

```powershell
python scripts\eval_gen.py --config configs\smoke.yaml --device cpu
```

`configs/default.yaml` 和 `configs/smoke.yaml` 是组合入口。正式实验参数只改 `configs/full/`，冒烟测试参数只改 `configs/smoke/`，不要在两套参数之间混用文件。

录制演示 GIF：

```powershell
python scripts\record_demo.py --config configs\default.yaml --policy random
python scripts\record_demo.py --config configs\default.yaml --policy greedy
python scripts\record_demo.py --config configs\default.yaml --policy ippo
```

## 输出文件

```text
outputs/logs/train_ippo.csv
outputs/logs/fixed_eval_rows.csv
outputs/logs/fixed_eval_summary.csv
outputs/logs/generalization_rows.csv
outputs/logs/generalization_summary.csv
outputs/logs/generalization_starts.csv
outputs/logs/generalization_verdict.txt
outputs/figures/ippo_training_curves.png
outputs/figures/warehouse_layout.png
outputs/figures/generalization_metrics.png
outputs/figures/ippo_final_frame.png
outputs/models/ippo_actor_critic.pt
outputs/models/ippo_actor_critic_final.pt
outputs/videos/random_policy_demo.gif
outputs/videos/greedy_policy_demo.gif
outputs/videos/ippo_policy_demo.gif
```

冒烟测试输出：

```text
outputs/smoke/logs/smoke_metrics.csv
outputs/smoke/logs/evaluate_ippo.csv
outputs/smoke/figures/smoke_final_state.png
outputs/smoke/logs/
outputs/smoke/figures/
outputs/smoke/models/
outputs/smoke/videos/
```

`ippo_actor_critic.pt` 是按完成率、碰撞和奖励筛选的最佳 checkpoint；

`ippo_actor_critic_final.pt` 是最后一回合权重。

`fixed_eval_*.csv` 是固定训练起点公平对比，包含 Random、Greedy、Greedy+shield、IPPO greedy+shield 和 IPPO sample+shield；该结果只代表训练分布内表现。

`diagnose_ippo.py` 的 `evaluate_ippo.csv` 是 checkpoint 快速诊断输出，正式实验目录中不再保留该文件；冒烟配置下会写入 `outputs/smoke/logs/evaluate_ippo.csv`。

随机起点泛化评估会自动排除训练配置中的固定起点，并输出均值、标准差、95% 置信区间和阈值判定。该结果是有限未见起点上的统计证据，不是对所有可能起点的数学证明。
