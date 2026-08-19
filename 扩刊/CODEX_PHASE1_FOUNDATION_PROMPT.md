# Codex 执行任务：SMC 原会议修复复现 Gate 1

请先完整阅读仓库中的：

1. `docs/superpowers/specs/2026-08-19-smc-original-reproduction-design.md`
2. `docs/superpowers/plans/2026-08-19-smc-original-reproduction-plan.md`

本轮严格执行计划中的 **Task 1–Task 5**。计划已经提供经过独立语法检查和单元测试的参考代码，请优先按计划逐文件实现，不要自行重写成另一套架构。

---

## 一、任务分类与边界

这是原 SMC 会议工作的基础修复与复现，不是方法升级。本轮只建立：

- 可安装环境；
- 旧代码不可变审计；
- typed schemas；
- 确定性随机流；
- 合成实例银行；
- 显式机器时间线；
- Weibull 可靠性数学；
- 最终指标；
- schedule validator。

本轮不得实现：

- 九条调度规则；
- FIFO/EDD/MRT/SPT/LPT；
- 环境 `step`；
- reward；
- DDQN/DQN/Q-learning/SARSA；
- 训练或论文图；
- GNN、PPO、PROCESS/PM/WAIT 动作；
- 外部数据集下载。

---

## 二、硬性约束

1. `code/`、`code1/`、`code2/` 全部冻结：不修改、不格式化、不移动、不删除。
2. 新代码全部放在 `original_repro/`。
3. 使用 Python 3.11。
4. 使用官方 `torch==2.10.0` CUDA 12.8 wheel。
5. 使用 TDD；每个 Task 独立提交一次，共 5 个提交。
6. 实例生成必须使用局部 `random.Random` 和 `numpy.random.RandomState`，不得改变全局 RNG 状态。
7. 实例保存必须使用确定性 JSON gzip；不得使用 pickle。
8. gzip 必须设置空 filename 与 `mtime=0`，使不同目标路径下相同实例的压缩字节完全一致。
9. 不静默吞异常。格式错误、schema 不支持、路径不存在均须给出明确异常。
10. 不伪造 Git SHA、硬件信息、测试数量或运行结果。
11. 如需改变计划中的接口、公式、路径或测试预期，先停止并报告理由，不自行扩大范围。

---

## 三、执行顺序

### Task 1：环境与包

完成：

- `pyproject.toml`；
- package initializer；
- README；
- RTX 5090 硬件验证；
- `.gitignore`；
- package import 测试。

安装命令以计划为准。硬件验证必须实际执行 CUDA 张量运算，不能只依赖 `nvidia-smi`。

### Task 2：冻结旧代码

完成：

- 设计文档复制到 spec 路径；
- `legacy_manifest.json`；
- 每个 legacy 文件的 size 与 SHA-256；
- audit 单元测试；
- 仓库级 manifest 一致性测试；
- `git diff --exit-code -- code code1 code2`。

### Task 3：schemas 与随机流

完成：

- `OperationSpec`、`JobSpec`、`MachineSpec`、`InstanceSpec`、`ScheduleInterval`；
- `set_global_seed`；
- `keyed_uniform`；
- common-random-number 测试。

### Task 4：实例与银行

完成：

- legacy-compatible generator；
- JSON-gzip round trip；
- 压缩字节稳定性；
- malformed/schema-version 测试；
- 540 个测试实例；
- 5×200 个训练实例；
- 两次完整生成 manifest byte-identical；
- manifest 共 1540 个条目。

### Task 5：时间线、可靠性、指标、validator

完成：

- `MachineTimeline`；
- `weibull_interval_failure_probability`；
- `ScheduleMetrics`；
- `validate_schedule`；
- overlap、arrival、precedence、eligibility、duplicate、missing-operation 等测试；
- 全量质量检查；
- Gate 1 后 legacy manifest 再验证。

---

## 四、额外边界测试

计划中已包含以下关键测试，不得删除或弱化：

1. 相邻区间不 overlap；
2. 完全相同的正时长区间 overlap；
3. internal gap 可被 `earliest_feasible_start` 使用；
4. ineligible machine 被报告；
5. operation 重复与缺失被报告；
6. PM 不计入 process time；
7. PM 计入 PM time；
8. `paper_trave` 使用真实 final completion；
9. 相同实例写入不同文件名时 gzip bytes 相同；
10. generator 不改变全局 Python/NumPy RNG；
11. 不支持的 schema version 明确失败；
12. 非 gzip 或损坏 JSON 明确失败。

再补充至少两项：

- `ScheduleInterval` 在负 machine id 时失败；
- `InstanceSpec` 在 operation processing-time vector 长度与机器数不一致时失败。

---

## 五、最终验证命令

Ubuntu/Linux：

```bash
cd original_repro
source .venv/bin/activate
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
python -m smc_repro.scripts.verify_hardware | tee hardware.json
```

实例银行重复性：

```bash
rm -rf /tmp/smc-bank-a /tmp/smc-bank-b
python -m smc_repro.scripts.build_instance_banks \
  --output-root /tmp/smc-bank-a \
  --test-repetitions 20 \
  --train-seeds 0 1 2 3 4 \
  --train-episodes 200 \
  --base-seed 20260819
python -m smc_repro.scripts.build_instance_banks \
  --output-root /tmp/smc-bank-b \
  --test-repetitions 20 \
  --train-seeds 0 1 2 3 4 \
  --train-episodes 200 \
  --base-seed 20260819
cmp /tmp/smc-bank-a/manifest.json /tmp/smc-bank-b/manifest.json
```

旧代码不可变性：

```bash
python -m smc_repro.scripts.audit_legacy_outputs \
  --repo-root .. \
  --output ../docs/audit/legacy_manifest_after_gate1.json
cmp ../docs/audit/legacy_manifest.json \
  ../docs/audit/legacy_manifest_after_gate1.json
cd ..
git diff --exit-code -- code code1 code2
```

---

## 六、完成后必须停止并报告

不得开始规则、环境或智能体开发。最终回答中完整给出：

```bash
git status --short
git log --oneline -5
git diff HEAD~5..HEAD --stat
git diff --exit-code -- code code1 code2
cd original_repro
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m smc_repro.scripts.verify_hardware
```

并报告：

- 测试总数、通过数、失败数；
- 两个 bank manifest 是否 byte-identical；
- manifest 条目数；
- manifest SHA-256；
- legacy manifest 文件数；
- 5 个 commit SHA 与 subject；
- 新增/修改文件清单；
- 与计划的任何偏差；
- 未解决问题。

随后停止，等待仓库上传和外部代码审阅。
