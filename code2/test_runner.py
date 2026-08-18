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


def run_scenario_plot(model_path=None, episodes=20):
    """在各测试场景上评估 DQN（Ours），绘制 makespan / Uave / TRave 三条误差棒曲线图。"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d, loaded_model_path = _load_test_model(model_path=model_path)
    d.L_test = episodes

    scenario_labels = [_scenario_label(p) for p in TEST_PARAMS]
    base_seed = 20260326

    mk_means, mk_stds = [], []
    u_means, u_stds = [], []
    tr_means, tr_stds = [], []

    for sc_idx, params in enumerate(TEST_PARAMS):
        print(f"\n=== Scenario {sc_idx + 1}/{len(TEST_PARAMS)}: {params} ===")
        seed = base_seed + sc_idx
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        total_tr, total_u, total_mk, _ = d.main(fixed_params=params, verbose=False)

        mk_means.append(float(np.mean(total_mk)))
        mk_stds.append(float(np.std(total_mk)))
        u_means.append(float(np.mean(total_u)))
        u_stds.append(float(np.std(total_u)))
        tr_means.append(float(np.mean(total_tr)))
        tr_stds.append(float(np.std(total_tr)))

        print(
            f"Ours: "
            f"makespan={mk_means[-1]:.2f}±{mk_stds[-1]:.2f}, "
            f"Uave={u_means[-1]:.4f}±{u_stds[-1]:.4f}, "
            f"TRave={tr_means[-1]:.4f}±{tr_stds[-1]:.4f}"
        )

    def _plot_metric(y_label, better_note, means, stds, filename):
        plt.figure(figsize=(16, 8))
        x = np.arange(len(TEST_PARAMS))
        plt.errorbar(
            x,
            means,
            yerr=stds,
            label="Ours",
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=3,
            color="#1f77b4",
            alpha=0.95,
        )
        plt.xticks(x, scenario_labels, rotation=45, ha="right")
        plt.xlabel("Scenarios (n-m-e)")
        plt.ylabel(f"{y_label} ({better_note})")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        fig_path = PROJECT_ROOT / filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"\n图已保存: {fig_path}")

    _plot_metric(
        y_label="Average Makespan",
        better_note="lower is better",
        means=mk_means,
        stds=mk_stds,
        filename=f"scenario_makespan_{timestamp}.png",
    )
    _plot_metric(
        y_label="Average Uave",
        better_note="higher is better",
        means=u_means,
        stds=u_stds,
        filename=f"scenario_uave_{timestamp}.png",
    )
    _plot_metric(
        y_label="Average TRave",
        better_note="lower is better",
        means=tr_means,
        stds=tr_stds,
        filename=f"scenario_trave_{timestamp}.png",
    )

    print(f"使用模型: {loaded_model_path}")
    print(f"测试总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    run_scenario_plot()
