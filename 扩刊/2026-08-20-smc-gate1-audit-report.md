# SMC 原会议复现 Gate 1 外部审计报告

**审计对象：** `rayyyyyyyyb/smc1` 的 GitHub `main` 分支  
**审计基线：** 最新可见提交 `437c254 docs: archive Gate 1 execution record`  
**审计日期：** 2026-08-20  
**下一阶段边界：** 完成所有正式实验开始前的代码、配置、实例银行、环境、规则、智能体与 smoke/preflight；不执行完整五种子 × 540 实例实验，不引入 GNN。

---

## 1. 最终判定

Gate 1 的总体判定为：

> **本机验收条件通过，但公共仓库的干净检出复现尚未通过。**

第一阶段已经完成了大部分原定目标：

- 建立了独立的 `扩刊/original_repro/` 包；
- 保持 `code/`、`code1/`、`code2/` 旧实现不改；
- 建立了 typed schema、确定性实例生成、JSON-gzip 序列化；
- 建立了显式时间线、Weibull 条件区间失效概率、指标和 validator；
- 在 RTX 5090 环境中留下了 100 个测试通过、Ruff/mypy/compileall 通过的执行记录；
- 两次生成的 1540 条实例银行 manifest 相同，记录 SHA-256 为
  `68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515`；
- Git 历史按环境、旧代码冻结、schema/seed、实例银行、指标核心等职责拆成了清晰提交。

但是，当前 `test_legacy_immutable.py` 会扫描本机 `code*` 目录中的**全部实际文件**，而提交到仓库的 `legacy_manifest.json` 同时记录了 `.idea/`、`__pycache__/` 和约 191 MB 的 `PM.txt` 等未跟踪/被忽略文件。一个新的 `git clone` 不会获得这些文件，因此该测试会在干净检出中必然失败。这是下一阶段开始前必须先关闭的 P0 问题。

因此，本阶段不能直接进入正式训练。正确顺序是：

```text
Gate 1.5：修复干净检出、随机键、metadata、指标/validator 等基础问题
        ↓
Gate 2：锁定三种复现 profile、状态、奖励、九规则和经典规则
        ↓
Gate 3：构造式调度环境、DL-DDQN/DQN/表格方法、checkpoint
        ↓
Preflight：本地实例银行物化、clean-worktree、端到端 smoke
        ↓
外部审阅通过后再写正式实验驱动与批量实验代码
```

---

## 2. 本次审计方法与边界

本次审计执行了以下工作：

1. 阅读 GitHub 提交历史和每个 Gate 1 功能提交；
2. 阅读 `扩刊/all.md` 中的 RED/GREEN、RTX 5090、实例银行和最终审查记录；
3. 阅读 `schemas.py`、`seeding.py`、`instance_generator.py`、`instance_io.py`、`timeline.py`、`reliability.py`、`metrics.py`、`validator.py`；
4. 阅读相关单元测试、旧代码 `code2/Job_shop.py`、`DQN.py`、`Maintain.py` 和原计划文档；
5. 从文件之间的数据流反推干净检出、随机键、指标和维护区间的边界行为。

受当前执行环境限制，本次无法从容器联网克隆仓库并独立运行 pytest。因此：

- “100 tests passed”“RTX 5090 + torch 2.10.0+cu128”等属于对仓库执行日志的核验；
- clean-clone manifest 问题、随机键碰撞、metadata 可变性和指标边界问题属于可由当前源码直接证明的问题；
- 下一阶段必须在用户连接的 5090 主机重新执行全部门禁，不能把本报告当作目标机器已经通过。

---

## 3. Gate 1 验收矩阵

| 验收项 | 状态 | 结论 |
|---|---|---|
| 独立 installable package | 通过 | `扩刊/original_repro` 结构清晰，README、pyproject、测试入口存在 |
| RTX 5090 CUDA smoke | 通过（日志证据） | 记录为 Python 3.11.16、torch 2.10.0+cu128、CUDA 12.8、`sm_120`、张量结果 14.0 |
| 旧受版本控制源码未修改 | 通过 | 提交历史和日志均记录 legacy diff 为 0 |
| 旧本机目录完整字节审计 | 通过（本机） | 本机 120 个实际文件两次 manifest 一致 |
| 干净 clone 可运行 immutability test | **不通过** | manifest 含未跟踪/忽略文件，clean clone 无法满足 |
| typed schema | 基本通过 | finite、PROCESS 正时长和 nominal 下限已在最终修复加入 |
| schema 真正不可变 | 需修复 | frozen dataclass 内仍保存可变 `dict` metadata |
| keyed common random numbers | 需修复 | 目前字符串分隔编码存在不同键元组映射到同一 payload 的碰撞 |
| 实例生成局部 RNG | 通过 | `random.Random` + `RandomState`，不改变全局生成器 |
| 确定性 JSON-gzip | 通过 | `filename=""`、`mtime=0`、排序 JSON |
| 1540 条固定实例银行 | 通过（manifest） | 540 测试 + 5×200 训练；gzip 本地存在但不进 Git |
| 新 clone 一键物化数据 | 需补 | README 只有 builder 描述，缺少依据提交 manifest 的物化与逐文件验证命令 |
| Weibull interval probability | 通过 | 数学实现正确，已处理非有限输入 |
| MachineTimeline | 基本通过 | 私有存储、只读 tuple、overlap 检查正确；应拒绝零时长请求 |
| true tardiness / paper TR / utilization | 基本通过 | 主公式已有；availability-adjusted denominator 需要 horizon 防御 |
| validator | 基本通过 | eligibility、arrival、precedence、overlap、nominal duration 已有；维护/换刀区间语义尚未验证 |
| 精确依赖环境可提交复现 | 需补 | 当前 `pyproject` 是范围约束；真实 lock 被 `.gitignore` 排除 |
| CI/clean-worktree gate | 缺失 | 当前没有从纯 Git 对象重建并测试的门禁 |

---

## 4. 必须先修复的问题

## 4.1 P0：legacy manifest 绑定本机未跟踪文件，干净 clone 必然失败

### 当前行为

当前审计脚本递归扫描：

```python
for dirname in ("code", "code1", "code2"):
    for path in sorted((root / dirname).rglob("*")):
        if path.is_file():
            ...
```

当前仓库级测试也用相同方法扫描所有实际文件，并要求和提交的 `legacy_manifest.json` 完全相同。

但 manifest 中包含：

```text
code/.idea/*
code/__pycache__/*
code/PM.txt                 # 191161849 bytes
code1/PM.txt                # 大型运行日志
其他本机未跟踪文件
```

这些文件被根 `.gitignore` 忽略或从未加入 Git。执行记录也明确区分了“Git 跟踪 35 个 legacy 文件”和“本机实际审计 120 个文件”。因此，新 clone 只能得到受版本控制的子集，`observed != manifest["files"]`。

### 根因

第一阶段把两个不同目标混成了一个 manifest：

1. **本地证据保全：** 审计最初机器上的所有实际文件；
2. **仓库可移植性：** 保证任何 clone 中受版本控制的 legacy 源码没有变化。

这两个目标不能由同一份“全目录严格相等”测试同时完成。

### 必须修改

保留两类 manifest：

```text
扩刊/docs/audit/legacy_local_full_manifest.json
扩刊/docs/audit/legacy_local_full_manifest_after_gate1.json
扩刊/docs/audit/legacy_tracked_manifest.json
```

含义：

- `legacy_local_full_manifest*.json`：历史证据，只证明 Gate 1 前后原机器的 120 个实际文件没变；不参与 clean clone pytest；
- `legacy_tracked_manifest.json`：只对 `git ls-files -- code code1 code2` 返回的文件做 SHA-256；参与持续测试；
- `test_legacy_immutable.py` 只比较 tracked manifest；
- 新增 clean-worktree 门禁，证明测试不依赖 ignored/untracked 文件。

这是正式实验前的第一项任务。

---

## 4.2 P1：`keyed_uniform()` 存在键编码碰撞

当前实现：

```python
payload = "|".join(map(str, (base_seed, *keys))).encode("utf-8")
```

以下两个不同键序列产生完全相同的 payload：

```python
keyed_uniform(7, "a|b", "c")
keyed_uniform(7, "a", "b|c")
```

此外：

```python
keyed_uniform(7, 1)
keyed_uniform(7, "1")
```

也无法区分类型。

当前实例银行不调用该函数，所以修复不会改变已提交的实例 manifest；但一旦环境用它产生策略无关故障流，碰撞会破坏 common-random-number 设计。

### 必须修改

采用类型标记 + 长度前缀编码：

```python
from __future__ import annotations

import hashlib
import math
import struct


def _encode_key(value: object) -> bytes:
    if value is None:
        tag = b"N"
        body = b""
    elif isinstance(value, bool):
        tag = b"B"
        body = b"1" if value else b"0"
    elif isinstance(value, int):
        tag = b"I"
        body = str(value).encode("ascii")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("floating-point random-stream keys must be finite")
        tag = b"F"
        body = value.hex().encode("ascii")
    elif isinstance(value, str):
        tag = b"S"
        body = value.encode("utf-8")
    else:
        raise TypeError(
            "random-stream keys must be None, bool, int, finite float, or str; "
            f"got {type(value).__name__}"
        )
    return tag + struct.pack(">Q", len(body)) + body


def keyed_uniform(base_seed: int, *keys: object) -> float:
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    payload = b"".join(_encode_key(value) for value in (base_seed, *keys))
    digest = hashlib.blake2b(payload, digest_size=8, person=b"smc-crn1").digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float(1 << 64)
```

必须增加碰撞回归测试，不能只测试“相同输入相同输出”。

---

## 4.3 P1：`PYTHONHASHSEED` 在解释器启动后设置，不能改变本进程哈希随机化

`set_global_seed()` 当前执行：

```python
os.environ["PYTHONHASHSEED"] = str(seed)
```

该赋值可以传给子进程，但 Python 当前进程的字符串/bytes 哈希种子在解释器启动时已经确定。不能把这行代码作为“当前进程 hash 完全固定”的证据。

### 必须修改

- 保留该环境变量赋值，用于子进程和运行元数据；
- 所有正式命令必须在启动 Python 之前设置：

PowerShell：

```powershell
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m smc_repro.scripts.preflight --config configs/paper_repro.yaml
```

Bash：

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m smc_repro.scripts.preflight --config configs/paper_repro.yaml
```

- `capture_environment.py` 必须记录两个环境变量的实际值；
- 不得在论文中声称不同 Python/PyTorch/CUDA 版本之间 bitwise identical，只能承诺固定软件栈内的确定性复现。

---

## 4.4 P1：release 只提交 manifest，缺少从新 clone 物化并验证 1540 个 gzip 的闭环

当前做法本身合理：1540 个生成数据文件没有写入 Git 历史，只提交了 manifest。但新用户目前只能看到“用 builder 生成数据”，缺少以下保证：

1. 是否按与 release 完全相同的参数生成；
2. 生成的 `manifest.json` 是否与提交的 reference manifest 字节一致；
3. 每个 gzip 是否存在且 SHA-256 与 manifest 一致；
4. 生成中途残缺时是否明确失败。

### 必须修改

新增：

```text
src/smc_repro/scripts/verify_instance_bank.py
src/smc_repro/scripts/capture_environment.py
```

固定数据准备命令：

```powershell
python -m smc_repro.scripts.build_instance_banks `
  --output-root artifacts/banks/materialized `
  --test-repetitions 20 `
  --train-seeds 0 1 2 3 4 `
  --train-episodes 200 `
  --base-seed 20260819

python -m smc_repro.scripts.verify_instance_bank `
  --reference artifacts/banks/release/manifest.json `
  --bank-root artifacts/banks/materialized `
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515
```

本阶段**不下载外部数据集**。原会议实验的“数据集”就是这套程序生成的固定实例银行。

---

## 4.5 P1：availability-adjusted utilization 缺少 horizon 防御

当前实现：

```python
available_capacity = machine_count * makespan - setup_time - pm_time - cm_time
```

如果输入 schedule 含有发生在最终工件完成之后的 PM/CM，或者某个非 PROCESS 区间跨越 `makespan`，其完整时长仍会被扣除。这可能导致：

- 分母被多扣；
- utilization 大于 1；
- 分母非正而静默返回 0；
- 环境错误被指标掩盖。

### 必须修改

双层防御：

1. validator 在完整 schedule 中要求 SETUP/PM/CM 的结束时刻不晚于全局最后 PROCESS 完成时刻；
2. metrics 仍将所有容量时长裁剪到 `[0, makespan]`。

辅助函数：

```python
def duration_within_horizon(interval: ScheduleInterval, horizon: float) -> float:
    left = max(0.0, interval.start)
    right = min(horizon, interval.end)
    return max(0.0, right - left)
```

并断言：

```python
0.0 <= standard_utilization <= 1.0 + 1e-9
0.0 <= availability_adjusted_utilization <= 1.0 + 1e-9
```

不能简单 `clip(0, 1)` 掩盖非法 schedule；越界时应抛出异常。

---

## 4.6 P1：validator 还没有检查 SETUP、PM、CM 的语义时长

当前 validator 只把非 PROCESS 区间当作占机区间检查 overlap，没有确认：

- SETUP duration 是否等于该机器 `setup_time`；
- PM duration 是否等于 `machine.pm_duration`；
- CM duration 是否等于 `machine.cm_duration`；
- setup_time 为 0 时是否错误生成零时长 SETUP；
- 完成最后一道工序后是否又添加没有任何调度意义的维护区间。

### 必须修改

完整 schedule 的 validator 增加：

```text
SETUP: duration == machine.setup_time；若 setup_time==0，不应创建区间
PM:    duration == machine.pm_duration
CM:    duration == machine.cm_duration
SETUP/PM/CM.end <= makespan
```

允许容差 `1e-9`。如果后续 profile 需要不同维护时长，必须在配置中显式覆盖并把 resolved duration 写入 interval metadata，不能直接放宽 validator。

---

## 4.7 P2：frozen dataclass 内部仍包含可变 metadata dict

当前：

```python
@dataclass(frozen=True)
class InstanceSpec:
    metadata: dict[str, Any] = field(default_factory=dict)
```

`frozen=True` 只阻止属性重新赋值，不能阻止：

```python
instance.metadata["machine_count"] = 999
```

这会使实例对象和已计算 hash/manifest 的语义脱离。

### 必须修改

将 metadata 限定为 JSON scalar，并在 `__post_init__` 做 defensive copy + `MappingProxyType`。`instance_io.py` 序列化时使用 `dict(instance.metadata)`。

本阶段 metadata 只需要保存字符串、整数、有限浮点、布尔和 `None`。嵌套复杂结构应放到独立 dataclass，不要继续扩大通用 `Any`。

---

## 4.8 P2：非 PROCESS interval 和 timeline search 仍允许零时长

当前 schema 只要求 PROCESS 正时长；timeline 的 `earliest_feasible_start()` 允许 `duration=0`。

虽然当前生成器的 setup、PM、CM 都为正数，但正式环境写入区间前应阻止 no-op 区间：

- 所有实际记录的 interval 必须 `end > start`；
- 若某 profile 的 setup time 为 0，环境直接不创建 SETUP interval；
- `earliest_feasible_start()` 要求 `duration > 0`。

这样能避免零时长区间扰乱排序、计数和事件日志。

---

## 4.9 P2：依赖范围已定义，但提交仓库中没有精确的运行环境快照

`pyproject.toml` 使用合理的版本范围，但范围会随时间解析到不同版本。`environment.lock.txt` 含真实环境，却被忽略，无法从 GitHub 复核。

### 必须修改

新增并提交机器可移植的：

```text
扩刊/docs/audit/environment_5090_resolved.json
```

内容由脚本生成，包括：

```text
Python/PyTorch/CUDA/GPU
pip list --format=json 的 name/version
平台信息
Git commit
bank manifest SHA
PYTHONHASHSEED
CUBLAS_WORKSPACE_CONFIG
```

不要提交带本机绝对路径的 editable freeze 行。正式实验 run metadata 还要复制这份信息并记录配置 hash。

---

## 4.10 P2：缺少 clean-worktree/CI 级别门禁

第一阶段所有测试都在包含本机 ignored 文件和既有 editable install 的工作目录运行，因此没有证明：

- 从纯 Git 对象检出后可以导入当前源码；
- 测试不依赖旧 editable 安装路径；
- 数据可由提交 manifest 重新生成；
- legacy test 不依赖本机日志。

下一阶段必须加入 `scripts/clean_worktree_gate.py` 或等价 PowerShell/Bash 命令，并将其纳入 preflight。

---

## 5. 当前做得好的部分

以下内容可以保留并直接作为下一阶段地基：

1. **独立目录策略正确。** 新实现没有继续复制修改 `code/code1/code2`。
2. **实例 schema 与确定性 gzip 设计正确。** JSON 可审计，`mtime=0` 避免路径/时间戳污染。
3. **实例银行规模和顺序明确。** 27 场景 × 20 测试实例，以及 5×200 训练实例已经固化。
4. **可靠性数学核心正确。** `weibull_interval_failure_probability()` 使用累计风险增量，而不是直接重复抽累计 CDF。
5. **时间线已从并行 Start/End/T 数组升级为完整 interval record。** 这是后续 SETUP/PM/CM 合法性的必要条件。
6. **指标明确区分 paper 和 standard 口径。** `paper_trave`、true tardiness、paper Uave、standard utilization 可以并列审计。
7. **最终审查修复有效。** finite、PROCESS 正时长、nominal duration 下限、timeline 只读暴露和 overlap 测试已经补齐。
8. **Git 提交职责清晰。** 下一轮适合继续按任务分提交并逐轮审查。

---

## 6. 下一阶段的准确范围

下一轮 Codex 应完成正式实验之前的所有依赖，但**不得开始完整实验**。

### 必须完成

```text
A. Gate 1.5 基础修复
   tracked/full legacy manifests
   clean clone gate
   collision-free keyed RNG
   immutable metadata
   interval/metrics/validator hardening
   exact environment snapshot
   bank materialization and verification

B. 复现协议固化
   legacy_snapshot.yaml
   paper_repro.yaml
   corrected_smc.yaml
   ambiguities.json
   strict config loader + config hash

C. 方法前置实现
   runtime state
   named observations
   reward functions
   legacy/paper/classical rules
   constructive scheduling environment
   DL-DDQN / DQN / Q-learning / SARSA
   complete checkpoint

D. 实验前 preflight
   all tests + Ruff + mypy + compileall
   clean worktree gate
   1540 bank verification
   3 profiles end-to-end tiny smoke
   all schedules pass validator
   checkpoint round trip and eval epsilon=0
   preflight_report.json
```

### 已锁定、不得由 Codex 静默改动的四项口径

1. **换刀/SETUP：** `legacy_snapshot` 与 `corrected_smc` 复刻源代码的 machine-specific tool-change；`paper_repro` 按论文“setup negligible”假设，不生成 SETUP 区间。
2. **紧急度：** 三个 profile 都统一为 `1=high, 2=medium, 3=low`。论文实验表中相反的文字属于内部冲突，必须记录在 `ambiguities.json`。
3. **交期：** 所有实例保存绝对 due date；动态工件使用 `arrival_time + (0.2 + 0.5*urgency)*estimated_work`。只有需要相对交期窗口时才减去 arrival。
4. **局部前插：** 本轮三个 profile 都采用 tail append；`paper_repro` 明确记录其与论文“earliest feasible idle slot”叙述的差异。健康依赖时长、PM/CM 和后续机器事件需要按时间重放，不能只把 PROCESS 区间回插后保留未来健康轨迹不变。

### 本轮禁止

- 不执行正式 200 episode × 5 seeds；
- 不执行所有方法 × 540 test instances；
- 不绘制论文结果图；
- 不做统计显著性汇总；
- 不引入 GNN/PPO/PyG/DGL；
- 不下载 Brandimarte、Hurink 等外部 benchmark；
- 不把 PM 改成 agent action。

正式实验代码与批量运行脚本应在下一次仓库审阅通过后再写。

---

## 7. 数据准备结论

原 SMC 会议代码使用随机生成的动态 FJSP 实例，不依赖外部数据集。因此本阶段的数据操作不是“联网下载”，而是：

1. 从提交的 reference manifest 重新生成固定实例；
2. 验证 manifest 字节和 SHA；
3. 验证 1540 个 gzip 逐文件 hash；
4. 保持 gzip 不进 Git；
5. 在每个正式 run metadata 记录 bank manifest SHA。

当前 reference：

```text
base_seed: 20260819
train seeds: 0 1 2 3 4
train episodes per seed: 200
test scenarios: 27
test repetitions per scenario: 20
instance count: 1540
manifest SHA-256:
68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515
```

外部标准 FJSP 数据只在后续 GNN 升级阶段下载；此时加入会改变“原会议复现”的问题分布，不应混入当前结果。

---

## 8. 下一次外部审阅的验收条件

Codex 完成下一阶段并上传后，必须同时满足：

```text
1. clean worktree 中全套 pytest 通过；
2. legacy_tracked_manifest 与 Git tracked code* 精确一致；
3. 原本 120 文件的 local full manifest 作为历史证据保留；
4. keyed_uniform 碰撞测试通过；
5. 1540 个实例银行由新 clone 成功物化并逐文件验证；
6. reference manifest SHA 保持 68a2fd...cd1515；
7. 三个 YAML profile 严格解析，未知 key 明确失败；
8. 九条 legacy rules、九条 paper rules、五条 classical rules 有手算测试；
9. 三个 profile 的 tiny instance 可完成且 validator 通过；
10. DL-DDQN/DQN/tabular smoke 无 NaN、无非法 schedule；
11. checkpoint 完整恢复，推理 epsilon 强制为 0；
12. no checkpoint 时明确 FileNotFoundError；
13. 不存在 code/code1/code2 diff；
14. 未开始正式批量实验；
15. 生成结构化 preflight_report.json 并列出所有真实命令和结果。
```

达到以上条件后，项目才真正进入“可以直接写并运行正式实验代码”的状态。
