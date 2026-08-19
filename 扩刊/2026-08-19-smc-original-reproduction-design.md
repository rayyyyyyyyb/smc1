# SMC 原会议代码修复与复现设计说明

**项目：** `rayyyyyyyyb/smc1`  
**审阅基线：** 2026-08-19 可见的 GitHub `main` 分支快照  
**适用阶段：** 只修复、审计并重跑原 SMC 会议工作；本阶段不引入 GNN，不更换九规则动作框架，不扩大论文问题定义。

---

## 1. 本阶段的最终目标

本阶段不是直接把旧代码“改到能跑”，而是产出一套可以审计、可以区分论文口径与修正口径、可以稳定重跑的原会议复现工程。最终必须同时回答三个问题：

1. **当前仓库代码实际做了什么？**
2. **按论文文字与公式，原会议实验应当如何实现？**
3. **在不改变 DL-DDQN + 九条复合规则这一算法家族的前提下，修复明显建模与实验问题后，结论是否仍成立？**

为防止三种口径混在一起，工程中固定使用三个 profile：

| Profile | 用途 | 是否可作为最终科学结论 |
|---|---|---|
| `legacy_snapshot` | 尽量复刻当前仓库行为，作为回归与错误定位基准 | 否，只用于审计 |
| `paper_repro` | 按论文公式、规则定义、基线和实验规模重建 | 可以用于“论文复现结果” |
| `corrected_smc` | 保留原算法框架，但修复真实拖期、时间线、随机流、公平比较等问题 | 可以用于“修复后结果” |

所有输出文件、图、checkpoint、结果表必须带 profile 名称。禁止把不同 profile 的数字放在同一列而不标注。

---

## 2. 范围边界

### 2.1 本阶段必须保留

- 双层 DDQN 的总体设想；
- 上层 2 个 reward-mode 动作；
- 下层 9 条复合派工规则动作；
- 动态新工件到达；
- 机器退化、换刀、PM 和 CM；
- 27 个测试场景：
  - 机器数 `{8, 12, 16}`；
  - 新增工件数 `{10, 20, 30}`；
  - 平均到达间隔 `{50, 100, 150}`；
- 原论文对照组：固定派工规则、DQN、Q-learning、SARSA、PM 消融。

### 2.2 本阶段明确不做

- 不引入 GNN；
- 不把 PM 改成智能体动作；
- 不改为 PPO；
- 不引入 PROCESS/PM/WAIT 可变动作空间；
- 不下载外部 FJSP 数据集；
- 不做新论文主方法；
- 不覆盖或删除 `code/`、`code1/`、`code2/`。

原论文实例完全由程序随机生成，因此本阶段**不需要下载数据集**。要做的是生成并固化训练/测试实例银行，而不是每个算法运行时临时随机生成不同实例。

---

## 3. 当前仓库与论文的关键不一致

下表是本阶段必须逐项关闭的问题。严重度含义：

- **P0：** 不修复则核心实验结论不可审计；
- **P1：** 会显著影响公平性或结果解释；
- **P2：** 工程复现与可维护性问题。

| 严重度 | 项目 | 当前仓库行为 | 论文口径/科学口径 | 处理方式 |
|---|---|---|---|---|
| P0 | `TRave` | 使用 `OPT + ETL` 与交期窗口比较，几乎不使用真实完成时刻 | 论文公式使用最终完成时刻 `C_i,ni`；科学口径也必须使用完成时刻 | `paper_repro` 实现论文 TR；`corrected_smc` 额外报告 true tardiness、tardy rate、weighted tardiness |
| P0 | 状态顺序 | 代码返回 `[Uave,Ustd,CRJave,CRJstd,TRave,TRstd]` | 论文写 `[CRJave,CRJstd,Uave,Ustd,TRave,TRstd]` | 使用命名 dataclass；profile 决定向量序列；禁止魔法下标 |
| P0 | 下层网络第 7 维 | 代码拼接上层最大 Q 值 | 论文文字称“chosen reward index 的 one-hot”，但又称 7 维，内部矛盾 | `paper_repro` 默认使用 1 维 `reward_id`；同时做 `max_q` 与 2 维 one-hot 敏感性实验 |
| P0 | A1 规则 | 选择 `OP_i/J_i` 最小 | 论文称 urgency-weighted completion rate | `paper_repro` 按论文公式；`legacy_snapshot` 保留旧行为 |
| P0 | A2 规则 | 公式与论文不同，且有括号/排序方向风险 | 论文：有逾期工件时最大化 tardiness/urgency；否则最小化 slack/remain-work | 建立独立、可单元测试的规则函数 |
| P0 | B1/B2 编号 | Rule1 类分支按最早完成选机，Rule2 类按最早开始选机 | 论文 B1=最早开始，B2=最早完成 | `paper_repro` 按论文编号；结果图必须使用明确规则名称而非仅 Rule 编号 |
| P0 | 论文基线 | 当前 `code/test_runner.py` 比较 Ours + Rule1~Rule9 | 论文图文称 FIFO、EDD、MRT、SPT、LPT | 实现真实的五个经典规则；九条复合规则另作为补充实验 |
| P0 | 旧结果文件 | 存在负 `TRave`，而当前代码已 `max(0, ...)` | 与当前代码不可能同时成立 | 旧 txt 只归档，禁止进入新汇总；每次运行写独立 run 目录 |
| P1 | 测试探索率 | `load_model()` 恢复 checkpoint 中 epsilon | 测试必须 `epsilon=0` | 推理加载后强制 0；加单元测试 |
| P1 | 缺 checkpoint | 只打印警告并使用随机初始化模型继续评估 | 评估必须失败退出 | `FileNotFoundError` |
| P1 | 初始状态 | 硬编码六维零向量 | 必须由环境状态计算 | `env.observe()` 生成初态 |
| P1 | PM/CM/换刀时间线 | 主要通过 `down_until` 表示，没有统一 interval 记录 | 所有占机事件应进入同一机器时间线 | `corrected_smc` 使用 `PROCESS/SETUP/PM/CM` interval |
| P1 | 局部前插 | `earliest_start` 已包含机器末尾时间，历史 gap 通常无法使用 | 应在完整时间线上找最早合法 gap | `corrected_smc` 重写 `earliest_feasible_start()` |
| P1 | Weibull 随机故障 | 把累计 CDF 反复当下一工序 Bernoulli 概率 | 应使用条件区间故障概率 | `corrected_smc` 使用 interval failure probability |
| P1 | 磨损 | 每工序随机下降 4–8，基本与持续时间无关 | 至少应有明确 profile；不可混淆 | `legacy_snapshot/paper_repro` 保留并记录；`corrected_smc` 使用有效年龄映射 |
| P1 | 公平随机性 | 所有随机事件共用全局 RNG，策略调用顺序改变后故障流也改变 | 各算法应使用同实例和算法无关故障流 | 实例银行 + keyed stochastic stream |
| P1 | PM 消融 | 同一 checkpoint 在 PM on/off 环境零样本测试 | 还应有分别训练的 PM/no-PM 公平消融 | 同时报 `zero_shot_toggle` 和 `retrained_ablation` |
| P1 | DQN 与表格法公平性 | 训练轮数、状态信息、奖励与随机种子不同 | 必须区分论文复现和 apples-to-apples 比较 | 两套协议分别输出 |
| P1 | 统计口径 | 单 checkpoint 在 20 个临时随机 episode 上报 std | 训练随机性与实例随机性应分层 | 至少 5 个训练种子；每场景 20 个共同实例；配对 CI |
| P2 | 路径 | `D:\code*` 硬编码 | 跨平台路径 | `Path(__file__)` + CLI `--output-root` |
| P2 | 输出 | 追加写 txt，容易混版本 | 长表 CSV/JSON + manifest | 每个 run 独立目录 |
| P2 | checkpoint | 未保存 optimizer、配置、随机状态 | 可恢复训练与审计 | 完整 checkpoint schema |
| P2 | 对象时间轴 | `Start`/`End` 分别排序，`T` 未同步 | 单一 interval record | dataclass 列表 |

---

## 4. 工程组织

不要直接继续在三个重复目录里同步修补。新建一个唯一权威实现：

```text
smc1/
  code/                       # 冻结，不修改
  code1/                      # 冻结，不修改
  code2/                      # 冻结，不修改
  original_repro/
    pyproject.toml
    README.md
    configs/
      legacy_snapshot.yaml
      paper_repro.yaml
      corrected_smc.yaml
    src/smc_repro/
      __init__.py
      config.py
      seeding.py
      schemas.py
      instance_generator.py
      instance_io.py
      timeline.py
      reliability.py
      metrics.py
      validator.py
      state_features.py
      rewards.py
      rules/
        __init__.py
        legacy_rules.py
        paper_rules.py
        classical_rules.py
      env.py
      agents/
        __init__.py
        networks.py
        replay.py
        dl_ddqn.py
        tabular.py
      experiment_io.py
      statistics.py
      plotting.py
    scripts/
      verify_hardware.py
      build_instance_banks.py
      train.py
      evaluate.py
      reproduce_fig2_rules.py
      reproduce_fig3_rl.py
      reproduce_fig4_pm.py
      aggregate.py
      audit_legacy_outputs.py
    tests/
      test_seeding.py
      test_instance_generator.py
      test_instance_io.py
      test_reliability.py
      test_timeline.py
      test_metrics.py
      test_validator.py
      test_state_features.py
      test_rewards.py
      test_paper_rules.py
      test_classical_rules.py
      test_env.py
      test_checkpoint.py
      test_common_random_numbers.py
    artifacts/
      banks/
      runs/
      summaries/
      figures/
      audit/
```

### 为什么必须新建目录

1. 保留原始证据，便于确认某个修复改变了什么；
2. 避免三个目录之间继续产生漂移；
3. 后续 GNN 升级可以把 `original_repro` 当可信基线；
4. 论文复现与科学修正可以通过配置切换，而不是复制整个代码树。

---

## 5. 三个 profile 的精确定义

## 5.1 `legacy_snapshot`

目的：尽量复刻当前 `code2` 的行为，用于回归测试。

固定内容：

- 状态顺序：`U, Ustd, CRJ, CRJstd, legacy_TR, TRstd`；
- 下层 context：上层最大 Q 值；
- 当前九规则实现；
- 当前 reward1 的 `+1/0/-1`；
- 当前累计 Weibull CDF Bernoulli；
- 当前按工序随机磨损；
- 当前阈值 PM；
- 当前构造式调度语义。

允许的工程修复：

- 路径可移植；
- checkpoint 不存在时失败；
- 测试 epsilon=0；
- 固化实例与种子；
- 输出结构化；
- 不再使用错位的 `Start/End/T` 并行数组，但调度结果应尽量等价。

不得把这个 profile 的结果写成论文最终结果。

## 5.2 `paper_repro`

目的：按论文公开描述复现。

固定内容：

- 状态顺序：`CRJave, CRJstd, Uave, Ustd, TRave, TRstd`；
- 上层网络：6→10→10→10→2；
- 下层网络默认：7→50×7→9；
- 下层第 7 维默认用 `reward_id` 的标量 0/1；
- 辅助敏感性：`max_q` 7 维；真正 one-hot 则改为 8 维；
- A1/A2/A3、B1/B2/B3 按论文公式；
- R1 按论文正文：改善为 +1，否则 -1；
- R2 按论文正文：+1/0/-1，保留 10% 容差；
- 最终 `TR_i = max(0, C_i-D_i)/(OPT_i+ETL_i)`，完整排程时 `ETL_i=0`；
- 经典基线 FIFO、EDD、MRT、SPT、LPT；
- 训练 200 episode；测试每场景 20 个实例；
- 27 个场景；
- PM 模型按论文可公开确认的阈值与恢复设定。

论文中不能唯一确定的部分必须写入 `ambiguities.json`，不得静默决定。

## 5.3 `corrected_smc`

目的：不改变“DL-DDQN 选择九规则 + 阈值 PM”算法身份，但修复建模与实验可信度。

相对于 `paper_repro` 增加：

- 全部占机事件进入统一时间线；
- 真正合法的局部前插；
- 条件区间故障概率；
- 有效年龄和健康指标统一；
- 真实 tardiness、weighted tardiness、tardy rate；
- 标准 utilization；
- 统一 schedule validator；
- 共同实例银行；
- 算法无关故障随机流；
- 多训练种子和配对统计；
- 相同奖励、相同训练预算的 apples-to-apples RL 基线；
- PM/no-PM 分别训练的公平消融。

---

## 6. 关键数学口径

## 6.1 真实完工与拖期

工件最终完成时刻：

\[
C_i=C_{i,n_i}.
\]

真实拖期：

\[
T_i=\max(0,C_i-D_i).
\]

平均拖期：

\[
\bar T=\frac{1}{N}\sum_i T_i.
\]

加权总拖期：

\[
TWT=\sum_i w_iT_i.
\]

延迟工件比例：

\[
\mathrm{TardyRate}=\frac{1}{N}\sum_i\mathbf 1[C_i>D_i].
\]

论文归一化拖期率：

\[
TR_i^{paper}=\frac{\max(0,C_i-D_i)}{\max(\epsilon,OPT_i+ETL_i)}.
\]

## 6.2 利用率

论文式机器平均利用率：

\[
U_{paper}=\frac{1}{M}\sum_k\frac{P_k}{C_k^{last}}.
\]

标准利用率：

\[
U_{std}=\frac{\sum_k P_k}{M C_{max}}.
\]

可用容量修正利用率：

\[
U_{avail}=\frac{\sum_k P_k}{M C_{max}-T_{PM}-T_{CM}-T_{setup}}.
\]

三个指标可以同时报告，但名称必须区分。

## 6.3 条件区间故障概率

机器已存活至有效年龄 `a`，候选加工持续 `d` 时：

\[
p_{fail}(a,d)=1-\exp\left(-\left[\left(\frac{a+d}{\eta}\right)^\beta-\left(\frac{a}{\eta}\right)^\beta\right]\right).
\]

`corrected_smc` 只使用该概率判断候选加工区间故障，不把累计 CDF 反复当单步概率。

## 6.4 共同随机数

不能只在每个方法前 `random.seed(s)`，因为不同策略会调用随机函数不同次数。故障样本必须通过稳定键生成：

```python
from __future__ import annotations

import hashlib


def keyed_uniform(base_seed: int, *keys: object) -> float:
    payload = "|".join(map(str, (base_seed, *keys))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float(1 << 64)
```

例如：

```python
u = keyed_uniform(
    scenario.failure_seed,
    "process_failure",
    instance.instance_id,
    job_id,
    op_id,
    machine_id,
)
```

这样同一工序—机器候选的故障随机数与算法调用顺序无关。

---

## 7. 实例银行

## 7.1 测试银行

27 个场景，每场景 20 个实例：

\[
27\times20=540\text{ instances}.
\]

每个实例保存：

```json
{
  "schema_version": 1,
  "instance_id": "test_m08_j10_e050_rep00",
  "instance_seed": 202608190000,
  "failure_seed": 202608290000,
  "machine_count": 8,
  "new_job_count": 10,
  "mean_interarrival": 50,
  "jobs": [],
  "machines": [],
  "generator_metadata": {}
}
```

## 7.2 训练银行

每个训练种子固定 200 个实例。最终建议 10 个训练种子：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

第一轮复现先跑 5 个种子：

```text
0, 1, 2, 3, 4
```

只有所有主趋势稳定后再补到 10 个种子。

## 7.3 生成器兼容规则

`legacy_snapshot` 与 `paper_repro` 应复刻原生成器的关键细节：

- 初始工件数 5；
- 每个工件 1–20 道工序；
- 每道工序至少 2 台、至多 `M-1` 台候选机器；
- 加工时间 1–50；
- 新工件到达间隔取指数分布后转整数；
- `EL∈{1,2,3}`；
- 交期公式 `A + (0.2 + 0.5*EL)*estimated_work`；
- 换刀时间 1–50；
- CM 时间 1–99。

注意：由交期公式可知 `EL=1` 交期最紧，因此统一解释为 `1=high urgency, 3=low urgency`。论文内部若有相反文字，应在 audit 中注明。

---

## 8. 评估协议

## 8.1 原论文式复现

输出三个主要图组：

1. **经典派工规则对比：** DL-DDQN、FIFO、EDD、MRT、SPT、LPT；
2. **强化学习对比：** DL-DDQN、DQN、Q-learning、SARSA；
3. **PM 消融：** PM on 与 PM off。

对每场景报告 20 个共同实例的均值和标准差。

## 8.2 修复后可信评估

- 每个深度方法至少 5 个独立训练种子；
- 每个 checkpoint 在同一 540 个测试实例上评估；
- 同一实例使用同一算法无关故障流；
- 报告：
  - 各训练种子均值；
  - 分层 bootstrap 95% CI；
  - 与 DL-DDQN 的配对差值；
  - win/tie/loss；
- 所有 schedule 必须通过 validator；
- 非法 schedule 不能仅跳过，必须使运行失败并保存最小复现输入。

## 8.3 PM 消融必须有两组

```text
A. zero_shot_toggle
   同一 checkpoint，测试时切换 PM on/off。

B. retrained_ablation
   分别在 PM on 和 PM off 环境训练同等预算的模型，再配对测试。
```

论文原图若只对应 A，复现报告必须明确；科学结论主要依据 B。

---

## 9. 输出格式

每次运行创建唯一目录：

```text
artifacts/runs/
  paper_repro__ddqn__seed-000__20260819T120000Z/
    config_resolved.yaml
    metadata.json
    checkpoint.pt
    train_episodes.csv
    validation.csv
    stdout.log
    stderr.log
```

`metadata.json` 至少包含：

```json
{
  "schema_version": 1,
  "profile": "paper_repro",
  "method": "ddqn",
  "train_seed": 0,
  "git_commit": "collected_by_git_rev_parse_HEAD",
  "python_version": "collected_from_sys_version",
  "torch_version": "collected_from_torch_version",
  "cuda_runtime": "collected_from_torch_version_cuda",
  "gpu_name": "collected_from_torch_cuda_get_device_name",
  "feature_order": ["CRJave", "CRJstd", "Uave", "Ustd", "TRave", "TRstd"],
  "lower_context": "reward_id_scalar",
  "checkpoint_sha256": "computed_from_checkpoint_bytes"
}
```

尖括号仅表示由脚本运行时自动采集真实值；Codex 不得手填假值。

新结果禁止追加到旧 `results_*.txt`。原 txt 移入：

```text
artifacts/audit/legacy_outputs/
```

并计算 SHA-256。

---

## 10. 5090 环境策略

原模型很小，主要耗时往往来自 Python 环境模拟而非 GPU 前向，因此：

- 只用单卡 `CUDA_VISIBLE_DEVICES=0`；
- 不使用 AMP；
- 不使用多 GPU；
- 不为追求吞吐修改算法语义；
- 优先确保确定性和审计性；
- 训练并行如后续需要，应按训练种子做多进程，而不是把一个小网络做 DDP。

建议使用 Python 3.11 与官方 CUDA 12.8 PyTorch wheel。安装后必须验证设备名、CUDA 可用性和一个真实 CUDA 张量运算。

---

## 11. 阶段门槛

### Gate 1：基础设施

- 新目录可安装；
- 旧目录无修改；
- 单元测试通过；
- 540 个测试实例可重复生成且哈希稳定；
- 手工 schedule 的指标计算正确；
- validator 能抓到重叠、前序和候选机器错误。

### Gate 2：规则与环境

- 九条 paper rules 有手算测试；
- 五个经典规则有手算测试；
- 三 profile 在同一小实例上产生预期差异；
- `legacy_snapshot` 与旧代码在固定无故障小实例上结果一致或差异有解释。

### Gate 3：智能体

- checkpoint 可完整恢复训练；
- 测试 epsilon 恒为 0；
- Double DQN target 有单元测试；
- 下层 context 三种解释可配置；
- 10 episode smoke run 无 NaN、无非法 schedule。

### Gate 4：完整复现

- 原论文三组图全部重跑；
- 5 个训练种子；
- 所有原始长表可追溯；
- 自动生成 paper/code/corrected 三口径差异报告；
- 结论只依据实际输出，不复用旧 txt 数字。

---

## 12. 本轮 Codex 只实施的内容

第一轮只完成 Gate 1，不实现环境决策、规则、DDQN 或整套训练。原因是：如果实例、时间线、指标和随机性地基有误，后面所有训练都必须重来。

第一轮交付后上传仓库，下一轮由我检查：

- 文件结构；
- 单元测试；
- 手工指标；
- 实例哈希；
- 时间线边界；
- 随机流是否真正与策略调用顺序解耦。

检查通过后再发第二轮 Codex 执行文档。
