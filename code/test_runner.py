import random
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from DQN import DQN
from project_paths import PROJECT_ROOT

# 27 个场景（Machine × Job_insert × E_ave）
# 注意：DQN.main(fixed_params) 的顺序是 (M_num, E_ave, Job_insert)
TOTAL_MACHINE = [8, 12, 16]
JOB_INSERT = [10, 20, 30]
E_AVE = [50, 100, 150]

TEST_PARAMS = [(m, e, j) for m in TOTAL_MACHINE for e in E_AVE for j in JOB_INSERT]


def _scenario_label(params):
    # n-m-e: Job_insert-Machine-E_ave（与你图中的横轴口径接近）
    return f"{params[2]}-{params[0]}-{params[1]}"


def _load_test_model(model_path=None):
    d = DQN(mode="test")
    if model_path and Path(model_path).exists():
        model_path = Path(model_path)
        d.load_model(str(model_path))
        return d, model_path

    model_files = [p.name for p in PROJECT_ROOT.glob("dqn_model_*.pth")]
    if model_files:
        latest_model = sorted(model_files)[-1]
        model_path = PROJECT_ROOT / latest_model
        d.load_model(str(model_path))
        return d, model_path

    print("警告: 没有找到训练好的模型，使用初始模型进行测试")
    return d, None


def run_testing(model_path=None):
    """常规测试（独立文件）"""
    print("\n开始测试（固定参数）...")
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d, model_path = _load_test_model(model_path=model_path)

    all_results = []

    for params in TEST_PARAMS:
        print(f"\n测试参数: Machine={params[0]}, E_ave={params[1]}, Job_insert={params[2]}")
        Total_tard, Total_uk_ave, Total_makespan, TR = d.main(fixed_params=params)

        TR_ave = round(sum(Total_tard) / d.L_test, 2)
        uk_ave = round(sum(Total_uk_ave) / d.L_test, 4)
        makespan_ave = round(sum(Total_makespan) / d.L_test, 2)

        std1 = round(np.sqrt(sum(np.square(ta - TR_ave) for ta in Total_tard) / d.L_test), 2)
        std2 = round(np.sqrt(sum(np.square(ua - uk_ave) for ua in Total_uk_ave) / d.L_test), 2)
        std3 = round(np.sqrt(sum(np.square(ma - makespan_ave) for ma in Total_makespan) / d.L_test), 2)

        result_line = (
            "指标(episode均值±标准差): "
            f"TR_ave={TR_ave}, TR_std={std1}, "
            f"uk_ave={uk_ave}, uk_std={std2}, "
            f"makespan_ave={makespan_ave}, makespan_std={std3}"
        )
        all_results.append((params, result_line))
        print(f"测试结果: {result_line}")

        with open(PROJECT_ROOT / "data_test.txt", "a", encoding="utf-8") as file:
            file.write(f"\n=== 测试详细数据 {timestamp} ===\n")
            file.write(f"测试参数: Machine:{params[0]}, E_ave:{params[1]}, Job_insert:{params[2]}\n")
            file.write(f"TRave：{Total_tard}\n")
            file.write(f"uk：{Total_uk_ave}\n")
            file.write(f"makespan：{Total_makespan}\n")
            file.write(f"rewards：{TR}\n")
            file.write("=" * 50 + "\n")

    with open(PROJECT_ROOT / "results_test.txt", "a", encoding="utf-8") as file:
        file.write(f"\n=== 测试结果汇总 {timestamp} ===\n")
        file.write(f"使用的模型: {model_path}\n")
        for params, result in all_results:
            file.write(f"参数(Machine:{params[0]}, E_ave:{params[1]}, Job_insert:{params[2]}): {result}\n")
        file.write("=" * 50 + "\n")

    end_time = time.time()
    print(f"测试完成，耗时: {end_time - start_time:.2f} 秒")


def run_rule_comparison_plot(model_path=None, episodes=20):
    """生成 Ours + Rule1~Rule9 对比图（3个目标：makespan / Uave / TRave）"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d, loaded_model_path = _load_test_model(model_path=model_path)
    d.L_test = episodes

    policies = [("Ours", None)] + [(f"Rule{i}", i - 1) for i in range(1, 10)]
    scenario_labels = [_scenario_label(p) for p in TEST_PARAMS]
    base_seed = 20260326

    mk_means = {name: [] for name, _ in policies}
    mk_stds = {name: [] for name, _ in policies}
    u_means = {name: [] for name, _ in policies}
    u_stds = {name: [] for name, _ in policies}
    tr_means = {name: [] for name, _ in policies}
    tr_stds = {name: [] for name, _ in policies}

    for sc_idx, params in enumerate(TEST_PARAMS):
        print(f"\n=== Scenario {sc_idx + 1}/{len(TEST_PARAMS)}: {params} ===")
        for policy_name, fixed_rule_idx in policies:
            # 同一 scenario 下，给所有策略同一随机种子，增强对比公平性
            seed = base_seed + sc_idx
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            total_tr, total_u, total_mk, _ = d.main(
                fixed_params=params,
                fixed_rule_idx=fixed_rule_idx,
                verbose=False,
            )
            mk_mean = float(np.mean(total_mk))
            mk_std = float(np.std(total_mk))
            u_mean = float(np.mean(total_u))
            u_std = float(np.std(total_u))
            tr_mean = float(np.mean(total_tr))
            tr_std = float(np.std(total_tr))

            mk_means[policy_name].append(mk_mean)
            mk_stds[policy_name].append(mk_std)
            u_means[policy_name].append(u_mean)
            u_stds[policy_name].append(u_std)
            tr_means[policy_name].append(tr_mean)
            tr_stds[policy_name].append(tr_std)

            print(
                f"{policy_name}: "
                f"makespan={mk_mean:.2f}±{mk_std:.2f}, "
                f"Uave={u_mean:.4f}±{u_std:.4f}, "
                f"TRave={tr_mean:.4f}±{tr_std:.4f}"
            )

    def _plot_metric(metric_name, y_label, better_note, means_by_policy, stds_by_policy, filename):
        plt.figure(figsize=(16, 8))
        x = np.arange(len(TEST_PARAMS))
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for idx, (policy_name, _) in enumerate(policies):
            plt.errorbar(
                x,
                means_by_policy[policy_name],
                yerr=stds_by_policy[policy_name],
                label=policy_name if policy_name != "Rule9" else "Rule9 (Random)",
                marker="o",
                markersize=3,
                linewidth=1.3,
                capsize=2,
                color=colors[idx % len(colors)],
                alpha=0.9,
            )
        plt.xticks(x, scenario_labels, rotation=45, ha="right")
        plt.xlabel("Scenarios (n-m-e)")
        plt.ylabel(f"{y_label} ({better_note})")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        fig_path = PROJECT_ROOT / filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"\n图已保存: {fig_path}")

    _plot_metric(
        metric_name="makespan",
        y_label="Average Makespan",
        better_note="lower is better",
        means_by_policy=mk_means,
        stds_by_policy=mk_stds,
        filename=f"rule_comparison_makespan_{timestamp}.png",
    )
    _plot_metric(
        metric_name="Uave",
        y_label="Average Uave",
        better_note="higher is better",
        means_by_policy=u_means,
        stds_by_policy=u_stds,
        filename=f"rule_comparison_uave_{timestamp}.png",
    )
    _plot_metric(
        metric_name="TRave",
        y_label="Average TRave",
        better_note="lower is better",
        means_by_policy=tr_means,
        stds_by_policy=tr_stds,
        filename=f"rule_comparison_trave_{timestamp}.png",
    )

    print(f"使用模型: {loaded_model_path}")
    print(f"测试总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    # 生成和示例类似的 Ours + Rule1~Rule9 对比图
    run_rule_comparison_plot()
