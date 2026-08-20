# Codex 执行任务：SMC 原会议复现——实验前就绪阶段

请在仓库 `rayyyyyyyyb/smc1` 的最新 `main` 基础上执行本阶段任务。先完整阅读：

1. 仓库根目录的 `2026-08-20-smc-gate1-audit-report.md`（Task 0 会将其原样复制为正式 spec）
2. `扩刊/docs/superpowers/plans/2026-08-20-smc-preexperiment-implementation-plan.md`
3. `扩刊/all.md`
4. 当前 `扩刊/original_repro/` 全部源码和测试
5. 仓库根 `code/`、`code1/`、`code2/` 中与规则、调度、维护、DQN、表格强化学习有关的源文件

本轮执行计划中的 **Task 0–Task 9**，然后运行完整 preflight 和 clean-worktree gate，提交执行证据并停止。

---

## 一、任务目标

把第一阶段 Gate 1 基础工程推进到“可以在下一轮直接编写正式实验驱动代码”的状态：

```text
修复干净 clone 和基础边界问题
        ↓
物化并验证原会议合成实例银行
        ↓
锁定 legacy_snapshot / paper_repro / corrected_smc 三种口径
        ↓
实现命名状态、奖励、九条规则和经典规则
        ↓
实现构造式调度环境
        ↓
实现 DL-DDQN、vanilla-DQN target、Q-learning、SARSA 和严格 checkpoint
        ↓
完成小规模 smoke、preflight、clean-worktree 证明
        ↓
停止，等待外部审阅
```

本轮不是正式实验阶段。不得生成或宣称论文结果。

---

## 二、已知第一阶段问题，必须关闭

### P0：clean clone 的 legacy manifest 失败

当前 `legacy_manifest.json` 包含本机未跟踪/忽略文件，而 `test_legacy_immutable.py` 扫描全部实际文件。干净 clone 缺少这些文件，测试会失败。

必须拆分：

```text
legacy_local_full_manifest.json
legacy_local_full_manifest_after_gate1.json
legacy_tracked_manifest.json
```

历史 full manifest 只作为证据；持续测试只使用 `git ls-files -- code code1 code2` 的 tracked manifest。

### P1：`keyed_uniform` 的分隔符和类型碰撞

必须改为类型标记 + 长度前缀编码，并增加：

```text
("a|b", "c") != ("a", "b|c")
1 != "1"
True != 1
```

的回归测试。

### P1：schema/validator/metrics 边界

必须完成：

```text
metadata 防御性复制且只读
所有记录区间为正时长
维护/换刀区间不能越过最终 PROCESS horizon
availability-adjusted utilization 仅使用 horizon 内停机
SETUP/PM/CM 持续时间与机器配置一致
```

### P1：数据物化和环境快照

原会议不需要下载外部数据集。必须从生成器物化 1540 个 JSON-gzip 实例并验证提交的 reference manifest：

```text
68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515
```

同时提交可移植的 RTX 5090/Python/PyTorch/依赖元数据快照，不提交绝对路径形式的 editable lock。

---

## 三、硬性范围

### 必须实现

- Task 0：tracked/local legacy audit 拆分和 clean-clone contract
- Task 1：随机键、metadata、正时长不变量
- Task 2：指标和 validator 加固
- Task 3：实例银行物化/验证与环境元数据
- Task 4：严格 profile 和 ambiguity register
- Task 5：runtime、命名 observation、奖励
- Task 6：legacy/paper 九规则和五个 classical+ECT 基线
- Task 7：profile-controlled 构造式环境
- Task 8：DL-DDQN、vanilla-DQN target、Q-learning、SARSA、严格 checkpoint
- Task 9：preflight 和 clean-worktree gate

### 严禁实现

- 正式 200 episode × 5 seed 训练
- 完整 540-instance × 多方法评估
- 批量实验调度脚本
- 汇总、bootstrap、显著性检验
- 论文图表
- GNN、PyG、DGL、PPO
- PROCESS/PM/WAIT 新动作空间
- 外部 benchmark 下载
- W&B、Ray、Hydra、Gymnasium
- 修改、移动、删除或格式化 `code/`、`code1/`、`code2/`

Smoke 训练上限：每个 profile/method 3–5 个 episode；评估最多每个 checkpoint 2 个实例。

---

## 四、工作流要求

1. 记录开始 HEAD、`git status --short`、`git log --oneline -8`。
2. 使用独立 worktree 或隔离分支；不能在存在无关修改的工作区直接开发。
3. 严格 TDD：每个 Task 先写失败测试并记录 RED，再最小实现并记录 GREEN。
4. Task 0–Task 9 各一个聚焦提交；最终允许一个只更新 `扩刊/all.md` 的证据提交。`all.md` 中记录的是该提交之前的 pre-log evidence；提交日志后必须在最终 HEAD 重新运行 preflight/clean-worktree，并把两个忽略报告及其独立 SHA-256 随下一次审阅请求提供。不要为了把包含当前 Git SHA 的最终 report hash 写回仓库而反复 amend，那个 hash 是自指的。
5. 所有偏离计划的接口、公式、文件路径、配置键必须先停止并说明；不得自行换架构。
6. 每个 Task 后运行 focused pytest；每两个 Task 至少运行一次 full pytest、Ruff、mypy、compileall。
7. 不伪造测试、GPU、manifest、SHA、commit 或耗时。
8. 所有运行记录持续追加到 `扩刊/all.md`。

---

## 五、环境和数据准备

使用第一阶段已经安装的 RTX 5090 Python 3.11 环境。不要重装另一个 torch。

正式命令启动前设置：

PowerShell：

```powershell
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

Bash：

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

物化数据：

```powershell
cd 扩刊\original_repro
Remove-Item -Recurse -Force artifacts\banks\materialized -ErrorAction SilentlyContinue
python -m smc_repro.scripts.build_instance_banks `
  --output-root artifacts\banks\materialized `
  --test-repetitions 20 `
  --train-seeds 0 1 2 3 4 `
  --train-episodes 200 `
  --base-seed 20260819
python -m smc_repro.scripts.verify_instance_bank `
  --reference artifacts\banks\release\manifest.json `
  --bank-root artifacts\banks\materialized `
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 `
  --report artifacts\preflight\bank_verification.json
```

验收必须是：

```text
expected_file_count = 1540
verified_file_count = 1540
ok = true
```

不得提交 1540 个 `.json.gz` 文件；只提交 reference manifest、生成/验证代码和 README 命令。

---

## 六、三种 profile 必须严格隔离

### `legacy_snapshot`

```text
状态顺序：U, Ustd, CRJ, CRJstd, TR, TRstd
初始网络状态：全零
TR feature：旧 workload pressure
U feature：paper_uave
上层 hidden：10×3
下层 hidden：50×7
下层 context：upper max-Q scalar
规则：source-compatible legacy rules
setup：source `change_cutter()`，显式 SETUP interval
故障：prestart cumulative Weibull CDF
磨损：每工序随机 4–8
local insertion：关闭/尾部追加
```

### `paper_repro`

```text
状态顺序：CRJ, CRJstd, U, Ustd, TR, TRstd
初始状态：环境计算
TR feature：旧 workload pressure（论文未完整定义 partial TR）
最终评价：真实完工时刻 paper TR + true tardiness
上层 hidden：10×2
下层 hidden：50×2
下层 context：reward-id scalar（解决 7D/one-hot 矛盾）
规则：论文 A1/A2/A3 × B1/B2/B3
setup：关闭（论文明确假设 setup time negligible）
故障/磨损：保持原会议模型
```

### `corrected_smc`

```text
状态顺序：论文顺序
初始状态：环境计算
TR feature：projected completion tardiness ratio
U feature：standard utilization
可靠性：conditional interval failure probability
磨损：effective age
规则：论文规则
setup：保留 source `change_cutter()` 并显式记录
local insertion：关闭/尾部追加
```

本阶段三个 profile 都禁止把新工序回填到已经发生过的历史 gap。原因是：一旦加工时间、PM/CM 和健康退化依赖时间顺序，回填会改变后续所有机器事件的年龄、故障概率和持续时间；没有完整的时间线重放引擎时，所谓“只修复 local insertion”会产生内部不一致。把论文与源码的 local-insertion 差异写入 `ambiguities.json`（A-009），真正的历史 gap 插入留到后续事件驱动升级。

不得把三种口径的结果、checkpoint 或配置混在同一名字下。

还必须在 `ambiguities.json` 中显式锁定：

```text
A-010 setup/tool-change：legacy=source_tool_change，paper=none，corrected=source_tool_change
A-011 urgency：统一 1=high、2=medium、3=low
A-012 due date：存储绝对 due date；动态作业使用 arrival + urgency-scaled estimated work
```

---

## 七、规则实现的关键锁定

### Source-compatible legacy

- A1：最小 completed-operation count ratio
- A2：逐字复刻源码的两个分支和括号
- A3：独立 policy stream 随机
- legacy B1：ECT（源码 Rule 1/4/7）
- legacy B2：EST（源码 Rule 2/5/8）
- B3：合法机器中独立 policy stream 随机

### Paper rules

- A1：`completion_ratio_by_work / (4-urgency)` 最小
- A2 有拖期：`(decision_time-due)/urgency` 最大
- A2 无拖期：`(due-processed_work)/remaining_nominal_work` 最小
- B1：EST
- B2：ECT
- B3：随机

### Classical

```text
FIFO+ECT
EDD+ECT
MRT+ECT
SPT+ECT
LPT+ECT
```

所有 tie 都按明确规则处理；随机选择不能使用全局 `random`。

---

## 八、环境实现的关键锁定

这是原会议**构造式/list-scheduling 环境**，不是新 GNN 阶段的事件驱动 SMDP。

每个动作的显式区间顺序：

```text
SETUP（满足 source change_cutter predicate 时）
PM（阈值触发）
CM（开工前抽样）
PROCESS
健康/年龄更新
```

`change_cutter` 条件必须同时考虑：

```text
该 job 上一步是否在另一台 machine
该 machine 上一步是否加工另一 job
```

legacy high-load branch 使用**各机器最新 PROCESS 结束时刻（source CTK）**的 90th percentile，不得使用含 SETUP/PM/CM 的完整 timeline availability，也不是 75th percentile。

不得声称 within-operation breakdown；原会议是 prestart CM delay。

所有环境随机流按用途分离：

```text
policy
failure_primary
failure_secondary
wear
cm_recovery
replay
```

故障、磨损和 CM 恢复等环境随机数必须按 `(namespace, instance_id, job_id, op_id, machine_id)` 键控，不得包含 `decision_index`，从而使不同算法即使以不同决策顺序到达同一工序—机器对，也使用同一个环境随机数。只有策略探索/随机规则的随机数允许额外包含 `decision_index`。

每一步必须是完整事务：先在临时时间线和临时 machine/job runtime 上计算、验证 SETUP/PM/CM/PROCESS 与所有健康/年龄/计数更新，全部通过后再一次性提交。失败不能留下 interval、health、age、usage、count、next-op 或 decision clock 的任何部分状态。

每个显式区间必须携带稳定、可审计的事件标识：

```text
<instance_id>:d<decision_index:06d>:<interval_type_lowercase>
```

并至少记录 `event_id`、`decision_index`、`selected_job_id`、`selected_op_id`。同一决策至多产生一个 SETUP、PM、CM 和 PROCESS 区间，因此该格式在本阶段唯一；不得使用 Python 对象地址、时间戳或随机 UUID。

---

## 九、智能体和 checkpoint 的关键锁定

### DL-DDQN

- 上层输出 2 个 reward mode Q
- 下层输出 9 个 rule Q
- 两个网络使用独立 Adam
- MSE loss
- replay capacity/batch/gamma/epsilon/target update 从 profile 读取
- 一个 epsilon Bernoulli 同时决定上/下层是否随机，复刻源码行为
- Double DQN：online 选，target 评
- vanilla DQN：target max
- 同一个 transition scalar reward 训练上下层，保留原论文方法身份

### Lower context

- `max_q_scalar` → 7D
- `reward_id_scalar` → 7D
- `reward_id_one_hot` → 8D，仅 sensitivity 能使用

所有网络初始化必须只由显式 `seed` 决定，且不能消耗或泄漏调用者的全局 torch RNG：在 `torch.random.fork_rng(...)` 上下文内设置 CPU/CUDA seed 并创建网络。测试必须证明：两次同 seed 初始化即使中间发生全局 torch 随机抽样，参数仍逐张量相同；不同 seed 至少一张参数不同。

Replay `Transition` 必须把输入状态数组复制为连续 `float32`、检查形状和有限值，并设为只读；禁止保存调用者可继续原地修改的 NumPy 引用。

### Checkpoint

必须保存：

```text
全部 online/target 网络
全部 optimizer
replay 内容与 RNG
agent RNG
torch CPU/CUDA RNG
epsilon/global step/decision count
profile/contract 及其 SHA
```

必须：

```text
缺 checkpoint -> FileNotFoundError
eval load -> epsilon 强制为 0
strict profile/contract mismatch -> 失败且不部分修改 agent
atomic save
torch.load(weights_only=True)
```

### Q-learning / SARSA

- 10×9 Q table
- 六状态特征 clip 后求均值，映射 0–9
- 支持 source first-argmax sensitivity
- 科学默认采用 random tie-breaking，避免初始 Rule-1 偏置
- 支持 legacy_joint/fixed_tardiness/fixed_utilization reward protocol

---

## 十、最终门禁

在 RTX 5090 主机执行：

```powershell
cd 扩刊\original_repro
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
python -m smc_repro.scripts.verify_hardware
python -m smc_repro.scripts.verify_instance_bank `
  --reference artifacts\banks\release\manifest.json `
  --bank-root artifacts\banks\materialized `
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 `
  --report artifacts\preflight\bank_verification.json
python -m smc_repro.scripts.preflight `
  --repo-root ..\.. `
  --bank-root artifacts\banks\materialized `
  --reference-manifest artifacts\banks\release\manifest.json `
  --environment-metadata ..\docs\audit\environment_5090_resolved.json `
  --output artifacts\preflight\preflight_report.json `
  --device cuda:0
$pythonExe = (Get-Command python).Source
python -m smc_repro.scripts.clean_worktree_gate `
  --repo-root ..\.. `
  --python-executable $pythonExe `
  --report artifacts\preflight\clean_worktree_report.json
```

最终必须满足：

```text
full pytest 全部通过
Ruff 通过
mypy strict 通过
compileall 通过
RTX 5090 CUDA tensor smoke 通过
bank 1540/1540 通过
preflight_report.status = passed
clean_worktree_report.status = passed
git status --short 为空
legacy diff 为 0
没有 materialized bank、checkpoint、preflight JSON 被 Git 跟踪
```

---

## 十一、完成后必须报告并停止

请输出：

```bash
git status --short
git log --oneline -15
git diff <starting-head>..HEAD --stat
git diff --exit-code -- code code1 code2
```

并逐项报告：

1. 开始/结束完整 Git SHA；
2. Task 0–Task 9 每个 commit SHA 与 subject；
3. 新增/修改/重命名文件清单；
4. 每个 Task 的 RED/GREEN 证据；
5. pytest 总数、通过、失败、耗时；
6. Ruff/mypy/compileall 完整结果；
7. GPU、Python、torch、CUDA、compute capability；
8. reference manifest SHA；
9. 1540/1540 bank 验证；
10. 写入 `all.md` 的 pre-log HEAD，以及 pre-log preflight scientific payload SHA；
11. 最终 evidence-commit HEAD，并在该 HEAD 重新运行后报告 final preflight JSON 与 final clean-worktree JSON 各自的文件 SHA-256；这些 final report 保持 ignored，不写回 Git；环境快照内的 Git SHA 仅表示快照采集时的 provenance，不要求等于最终 HEAD，最终运行契约另行记录当前 SHA；
12. smoke episode 数、checkpoint round-trip 数、epsilon=0 证明；
13. legacy tracked/local manifest 文件数和 SHA；
14. 与指导计划的任何偏差及其原因；
15. 未解决问题；
16. 两份 ignored JSON 报告的完整内容或可访问路径。

完成以上内容后停止。不得开始正式实验代码、完整训练、统计分析或 GNN 升级。
