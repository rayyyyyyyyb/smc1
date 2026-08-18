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


def run_testing(model_path=None, enable_preventive_maintenance=True):
    """常规测试（独立文件）。enable_preventive_maintenance=False 为关闭预防性维护的对比实验。"""
    print("\n开始测试（固定参数）...")
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d, model_path = _load_test_model(model_path=model_path)

    all_results = []

    for params in TEST_PARAMS:
        print(f"\n测试参数: Machine={params[0]}, E_ave={params[1]}, Job_insert={params[2]}")
        Total_tard, Total_uk_ave, Total_makespan, TR = d.main(
            fixed_params=params,
            enable_preventive_maintenance=enable_preventive_maintenance,
        )

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
        file.write(f"预防性维护: {'开启' if enable_preventive_maintenance else '关闭'}\n")
        file.write(f"使用的模型: {model_path}\n")
        for params, result in all_results:
            file.write(f"参数(Machine:{params[0]}, E_ave:{params[1]}, Job_insert:{params[2]}): {result}\n")
        file.write("=" * 50 + "\n")

    end_time = time.time()
    print(f"测试完成，耗时: {end_time - start_time:.2f} 秒")


def _collect_metrics_for_pm_setting(d, enable_preventive_maintenance, policy_label):
    """遍历 TEST_PARAMS，返回各指标均值/标准差序列。"""
    scenario_labels = [_scenario_label(p) for p in TEST_PARAMS]
    base_seed = 20260326
    mk_means, mk_stds = [], []
    u_means, u_stds = [], []
    tr_means, tr_stds = [], []

    for sc_idx, params in enumerate(TEST_PARAMS):
        print(f"\n=== [{policy_label}] Scenario {sc_idx + 1}/{len(TEST_PARAMS)}: {params} ===")
        seed = base_seed + sc_idx
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        total_tr, total_u, total_mk, _ = d.main(
            fixed_params=params,
            verbose=False,
            enable_preventive_maintenance=enable_preventive_maintenance,
        )
        mk_mean = float(np.mean(total_mk))
        mk_std = float(np.std(total_mk))
        u_mean = float(np.mean(total_u))
        u_std = float(np.std(total_u))
        tr_mean = float(np.mean(total_tr))
        tr_std = float(np.std(total_tr))

        mk_means.append(mk_mean)
        mk_stds.append(mk_std)
        u_means.append(u_mean)
        u_stds.append(u_std)
        tr_means.append(tr_mean)
        tr_stds.append(tr_std)

        print(
            f"{policy_label}: "
            f"makespan={mk_mean:.2f}±{mk_std:.2f}, "
            f"Uave={u_mean:.4f}±{u_std:.4f}, "
            f"TRave={tr_mean:.4f}±{tr_std:.4f}"
        )

    return {
        "scenario_labels": scenario_labels,
        "mk_means": mk_means,
        "mk_stds": mk_stds,
        "u_means": u_means,
        "u_stds": u_stds,
        "tr_means": tr_means,
        "tr_stds": tr_stds,
    }


def _plot_metric_multi(series_list, y_label, better_note, scenario_labels, filename):
    """同一指标多条曲线：series_list 为 [(label, means, stds, color), ...]"""
    plt.figure(figsize=(16, 8))
    x = np.arange(len(TEST_PARAMS))
    for label, means, stds, color in series_list:
        plt.errorbar(
            x,
            means,
            yerr=stds,
            label=label,
            marker="o",
            markersize=3,
            linewidth=1.3,
            capsize=2,
            color=color,
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


def run_metrics_plot(
    model_path=None,
    episodes=20,
    enable_preventive_maintenance=True,
    timestamp=None,
):
    """单条曲线：按场景绘制 DQN 的 makespan / Uave / TRave（一种 PM 设置）。"""
    start_time = time.time()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d, loaded_model_path = _load_test_model(model_path=model_path)
    d.L_test = episodes

    policy_label = "Ours (with PM)" if enable_preventive_maintenance else "Ours (no PM)"
    color = "#1f77b4" if enable_preventive_maintenance else "#ff7f0e"
    data = _collect_metrics_for_pm_setting(d, enable_preventive_maintenance, policy_label)
    sl = data["scenario_labels"]

    suffix = "pm" if enable_preventive_maintenance else "no_pm"
    _plot_metric_multi(
        [(policy_label, data["mk_means"], data["mk_stds"], color)],
        "Average Makespan",
        "lower is better",
        sl,
        f"dqn_makespan_{timestamp}_{suffix}.png",
    )
    _plot_metric_multi(
        [(policy_label, data["u_means"], data["u_stds"], color)],
        "Average Uave",
        "higher is better",
        sl,
        f"dqn_uave_{timestamp}_{suffix}.png",
    )
    _plot_metric_multi(
        [(policy_label, data["tr_means"], data["tr_stds"], color)],
        "Average TRave",
        "lower is better",
        sl,
        f"dqn_trave_{timestamp}_{suffix}.png",
    )

    print(f"使用模型: {loaded_model_path}")
    print(f"测试总耗时: {time.time() - start_time:.2f} 秒")


# 有 PM / 无 PM 对比用的配色（同图不同色）
_PM_COMPARE_COLORS = {"with_pm": "#1f77b4", "no_pm": "#ff7f0e"}


def run_pm_ablation_plots(model_path=None, episodes=20):
    """同一指标一张图：有预防性维护与无预防性维护两条曲线，不同颜色与图例。"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d, loaded_model_path = _load_test_model(model_path=model_path)
    d.L_test = episodes

    print(">>> 评估：有预防性维护 (with PM)")
    data_pm = _collect_metrics_for_pm_setting(d, True, "Ours (with PM)")
    print("\n>>> 评估：无预防性维护 (no PM)")
    data_no = _collect_metrics_for_pm_setting(d, False, "Ours (no PM)")

    sl = data_pm["scenario_labels"]
    series_mk = [
        ("Ours (with PM)", data_pm["mk_means"], data_pm["mk_stds"], _PM_COMPARE_COLORS["with_pm"]),
        ("Ours (no PM)", data_no["mk_means"], data_no["mk_stds"], _PM_COMPARE_COLORS["no_pm"]),
    ]
    series_u = [
        ("Ours (with PM)", data_pm["u_means"], data_pm["u_stds"], _PM_COMPARE_COLORS["with_pm"]),
        ("Ours (no PM)", data_no["u_means"], data_no["u_stds"], _PM_COMPARE_COLORS["no_pm"]),
    ]
    series_tr = [
        ("Ours (with PM)", data_pm["tr_means"], data_pm["tr_stds"], _PM_COMPARE_COLORS["with_pm"]),
        ("Ours (no PM)", data_no["tr_means"], data_no["tr_stds"], _PM_COMPARE_COLORS["no_pm"]),
    ]

    _plot_metric_multi(series_mk, "Average Makespan", "lower is better", sl, f"dqn_makespan_{timestamp}.png")
    _plot_metric_multi(series_u, "Average Uave", "higher is better", sl, f"dqn_uave_{timestamp}.png")
    _plot_metric_multi(series_tr, "Average TRave", "lower is better", sl, f"dqn_trave_{timestamp}.png")

    print(f"使用模型: {loaded_model_path}")
    print(f"测试总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    run_pm_ablation_plots()
