import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from DQN import DQN
from Job_shop import Situation
from project_paths import PROJECT_ROOT

from state_discretization import discretize_state_from_features
from tabular_rl import TabularConfig, TabularQLearning, TabularSARSA


# 27 个场景（Machine × Job_insert × E_ave）
TOTAL_MACHINE = [8, 12, 16]
JOB_INSERT = [10, 20, 30]
E_AVE = [50, 100, 150]
TEST_PARAMS = [(m, e, j) for m in TOTAL_MACHINE for e in E_AVE for j in JOB_INSERT]


def _scenario_label(params):
    # n-m-e: Job_insert-Machine-E_ave（与你图中的横轴口径接近）
    return f"{params[2]}-{params[0]}-{params[1]}"


def _load_test_model(model_path=None):
    d = DQN(mode="test")
    d.L_test = 1  # 仅占位，实际由调用方覆盖
    if model_path and Path(model_path).exists():
        d.load_model(str(model_path))
        return d, Path(model_path)

    model_files = [p.name for p in PROJECT_ROOT.glob("dqn_model_*.pth")]
    if model_files:
        latest_model = sorted(model_files)[-1]
        model_path = PROJECT_ROOT / latest_model
        d.load_model(str(model_path))
        return d, model_path

    print("警告: 没有找到训练好的模型，DQN 部分将使用初始模型。")
    return d, None


def _compute_episode_metrics(Sit: Situation):
    total_TR = 0.0
    makespan = 0.0

    M_num = Sit.M_num
    uk_ave = sum(Sit.UK) / M_num

    Job = Sit.Jobs
    for Ji in range(len(Job)):
        end_time = max(Job[Ji].End) if Job[Ji].End else 0.0
        makespan = max(makespan, end_time)

        OPT_i = sum(Job[Ji].T)
        ETL_i = 0.0
        for op_idx in range(Sit.OP[Ji], Sit.J[Ji]):
            pt_list = [pt for pt in Sit.Processing_time[Ji][op_idx] if pt != -1 and pt > 0]
            if pt_list:
                ETL_i += sum(pt_list) / len(pt_list)

        denom = OPT_i + ETL_i
        if denom > 0:
            DDL_i = float(Sit.D[Ji] - Sit.Ai[Ji])
            TR_i = max(0.0, (denom - DDL_i) / denom)
        else:
            TR_i = 0.0
        total_TR += TR_i

    TR_ave = total_TR / len(Job) if Job else 0.0
    return round(makespan, 2), round(uk_ave, 4), round(TR_ave, 6)


def _make_env_and_run_episode(
    agent,
    *,
    reward_mode: int,
    op_rule_mapping,
    d_gen: DQN,
    M_num_fixed: int,
    E_ave_fixed: int,
    New_insert_fixed: int,
    epsilon: float,
):
    """
    reward_mode:
      1 => Sit.reward1（基于 TRave）
      2 => Sit.reward2（基于 Uave）
      3 => Sit.reward3（Algorithm 5，TR 与 U 联合；仅 Q-learning/SARSA 使用）
    """
    Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = d_gen.Instance_Generator(
        M_num_fixed, E_ave_fixed, New_insert_fixed, mode="test"
    )
    Sit = Situation(
        J_num,
        M_num,
        O_num,
        J,
        Processing_time,
        D,
        A,
        Change_cutter_time,
        Repair_time,
        EL,
    )

    # 与 DQN.main 保持一致：初始 obs 用全 0
    prev_features = [0.0] * 6

    state_idx = discretize_state_from_features(prev_features)
    total_reward = 0.0

    if hasattr(agent, "q") and agent.q.shape[0] != 10:
        raise ValueError("Q-learning/SARSA 需要 num_states=10 的表格状态空间。")

    # 逐步调度
    s = state_idx
    a = agent.select_action(s, epsilon=epsilon)
    for op_idx in range(O_num):
        at_trans = op_rule_mapping[a]()
        Sit.scheduling(at_trans)

        next_features = list(Sit.Features())
        done = (op_idx == O_num - 1)

        if reward_mode == 1:
            r = Sit.reward1(prev_features[4], next_features[4])
        elif reward_mode == 2:
            r = Sit.reward2(prev_features[0], next_features[0])
        elif reward_mode == 3:
            r = Sit.reward3(
                prev_features[4],
                next_features[4],
                prev_features[0],
                next_features[0],
            )
        else:
            raise ValueError(f"reward_mode 必须是 1、2 或 3，当前为 {reward_mode}")

        next_state = discretize_state_from_features(next_features)

        if isinstance(agent, TabularSARSA):
            if done:
                a_next = 0
                agent.update(s, a, float(r), next_state, a_next, done=True)
            else:
                a_next = agent.select_action(next_state, epsilon=epsilon)
                agent.update(s, a, float(r), next_state, a_next, done=False)
                a = a_next
        else:
            agent.update(s, a, float(r), next_state, done=done)

        total_reward += float(r)
        prev_features = next_features
        s = next_state

        if done:
            break

    makespan, uk_ave, TR_ave = _compute_episode_metrics(Sit)
    return makespan, uk_ave, TR_ave, total_reward


def train_tabular_q_learning(
    *,
    train_episodes: int = 200,
    reward_mode: int = 1,
    q_config: TabularConfig | None = None,
):
    q_config = q_config or TabularConfig()
    agent = TabularQLearning(q_config)

    # 复用 DQN 的实例生成器（环境动态保持一致）
    d_gen = DQN(mode="train")
    rule_mapping = None  # delay

    for ep in range(train_episodes):
        seed = q_config.seed + ep
        random.seed(seed)
        np.random.seed(seed)

        # 训练实例使用随机范围（与 DQN 的训练分布一致）
        Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = d_gen.Instance_Generator(
            mode="train"
        )
        Sit = Situation(
            J_num,
            M_num,
            O_num,
            J,
            Processing_time,
            D,
            A,
            Change_cutter_time,
            Repair_time,
            EL,
        )

        prev_features = [0.0] * 6
        s = discretize_state_from_features(prev_features)
        done = False

        rule_mapping = [
            Sit.rule1,
            Sit.rule2,
            Sit.rule3,
            Sit.rule4,
            Sit.rule5,
            Sit.rule6,
            Sit.rule7,
            Sit.rule8,
            Sit.rule9,
        ]

        a = agent.select_action(s)
        for op_idx in range(O_num):
            at_trans = rule_mapping[a]()
            Sit.scheduling(at_trans)

            next_features = list(Sit.Features())
            done = (op_idx == O_num - 1)

            if reward_mode == 1:
                r = Sit.reward1(prev_features[4], next_features[4])
            elif reward_mode == 2:
                r = Sit.reward2(prev_features[0], next_features[0])
            elif reward_mode == 3:
                r = Sit.reward3(
                    prev_features[4],
                    next_features[4],
                    prev_features[0],
                    next_features[0],
                )
            else:
                raise ValueError(f"reward_mode 必须是 1、2 或 3，当前为 {reward_mode}")

            s_next = discretize_state_from_features(next_features)
            agent.update(s, a, float(r), s_next, done=done)

            prev_features = next_features
            s = s_next
            if done:
                break
            a = agent.select_action(s)

        agent.decay_epsilon()

    return agent


def train_tabular_sarsa(
    *,
    train_episodes: int = 200,
    reward_mode: int = 1,
    q_config: TabularConfig | None = None,
):
    q_config = q_config or TabularConfig()
    agent = TabularSARSA(q_config)

    d_gen = DQN(mode="train")

    for ep in range(train_episodes):
        seed = q_config.seed + ep
        random.seed(seed)
        np.random.seed(seed)

        Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = d_gen.Instance_Generator(
            mode="train"
        )
        Sit = Situation(
            J_num,
            M_num,
            O_num,
            J,
            Processing_time,
            D,
            A,
            Change_cutter_time,
            Repair_time,
            EL,
        )

        prev_features = [0.0] * 6
        s = discretize_state_from_features(prev_features)

        rule_mapping = [
            Sit.rule1,
            Sit.rule2,
            Sit.rule3,
            Sit.rule4,
            Sit.rule5,
            Sit.rule6,
            Sit.rule7,
            Sit.rule8,
            Sit.rule9,
        ]

        epsilon = agent.cfg.epsilon
        a = agent.select_action(s, epsilon=epsilon)

        for op_idx in range(O_num):
            at_trans = rule_mapping[a]()
            Sit.scheduling(at_trans)

            next_features = list(Sit.Features())
            done = (op_idx == O_num - 1)

            if reward_mode == 1:
                r = Sit.reward1(prev_features[4], next_features[4])
            elif reward_mode == 2:
                r = Sit.reward2(prev_features[0], next_features[0])
            elif reward_mode == 3:
                r = Sit.reward3(
                    prev_features[4],
                    next_features[4],
                    prev_features[0],
                    next_features[0],
                )
            else:
                raise ValueError(f"reward_mode 必须是 1、2 或 3，当前为 {reward_mode}")

            s_next = discretize_state_from_features(next_features)

            if done:
                a_next = 0
                agent.update(s, a, float(r), s_next, a_next, done=True)
                break

            a_next = agent.select_action(s_next, epsilon=epsilon)
            agent.update(s, a, float(r), s_next, a_next, done=False)

            prev_features = next_features
            s = s_next
            a = a_next

        agent.decay_epsilon()

    return agent


def evaluate_tabular_agent_on_scenarios(
    *,
    agent,
    reward_mode: int,
    eval_episodes: int = 20,
    seed_base: int = 20260326,
):
    """
    返回：
      makespan_means, makespan_stds, u_means, u_stds, tr_means, tr_stds
    """
    d_gen = DQN(mode="train")  # 仅用于 Instance_Generator

    rule_mapping_factory = [
        lambda Sit: Sit.rule1(),
        lambda Sit: Sit.rule2(),
        lambda Sit: Sit.rule3(),
        lambda Sit: Sit.rule4(),
        lambda Sit: Sit.rule5(),
        lambda Sit: Sit.rule6(),
        lambda Sit: Sit.rule7(),
        lambda Sit: Sit.rule8(),
        lambda Sit: Sit.rule9(),
    ]

    mk_means, mk_stds = [], []
    u_means, u_stds = [], []
    tr_means, tr_stds = [], []

    for sc_idx, (m, e, j) in enumerate(TEST_PARAMS):
        mk_list, u_list, tr_list = [], [], []
        for ep in range(eval_episodes):
            seed = seed_base + sc_idx * 1000 + ep
            random.seed(seed)
            np.random.seed(seed)

            # 固定实例：直接在 _make_env_and_run_episode 里生成
            # 这里的 op_rule_mapping 依赖 Sit 实例方法，因此在运行时内部再绑定
            Processing_time, A, D, M_num, Op_num, J_dict, O_num, J_num, Change_cutter_time, Repair_time, EL = d_gen.Instance_Generator(
                m, e, j, mode="test"
            )
            Sit = Situation(
                J_num,
                M_num,
                O_num,
                J_dict,
                Processing_time,
                D,
                A,
                Change_cutter_time,
                Repair_time,
                EL,
            )
            op_rule_mapping = [
                Sit.rule1,
                Sit.rule2,
                Sit.rule3,
                Sit.rule4,
                Sit.rule5,
                Sit.rule6,
                Sit.rule7,
                Sit.rule8,
                Sit.rule9,
            ]

            prev_features = [0.0] * 6
            s = discretize_state_from_features(prev_features)
            epsilon = 0.0
            a = agent.select_action(s, epsilon=epsilon)

            for op_idx in range(O_num):
                at_trans = op_rule_mapping[a]()
                Sit.scheduling(at_trans)

                next_features = list(Sit.Features())
                done = (op_idx == O_num - 1)

                r = 0.0  # evaluation 不再更新 Q
                s_next = discretize_state_from_features(next_features)

                if done:
                    prev_features = next_features
                    s = s_next
                    break
                prev_features = next_features
                s = s_next
                a = agent.select_action(s, epsilon=epsilon)

            makespan, uk_ave, TR_ave = _compute_episode_metrics(Sit)
            mk_list.append(makespan)
            u_list.append(uk_ave)
            tr_list.append(TR_ave)

        mk_means.append(float(np.mean(mk_list)))
        mk_stds.append(float(np.std(mk_list)))
        u_means.append(float(np.mean(u_list)))
        u_stds.append(float(np.std(u_list)))
        tr_means.append(float(np.mean(tr_list)))
        tr_stds.append(float(np.std(tr_list)))

        print(
            f"Scenario {sc_idx+1}/{len(TEST_PARAMS)} done. "
            f"mk={mk_means[-1]:.2f}±{mk_stds[-1]:.2f}, "
            f"u={u_means[-1]:.4f}±{u_stds[-1]:.4f}, "
            f"tr={tr_means[-1]:.4f}±{tr_stds[-1]:.4f}"
        )

    return mk_means, mk_stds, u_means, u_stds, tr_means, tr_stds


def _resolve_checkpoint(path) -> Path | None:
    """支持项目根目录下的相对文件名或绝对路径。"""
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p if p.exists() else None


def run_dqn_and_tabular_comparison(
    *,
    dqn_model_path=None,
    ddqn_model_path=None,
    reward_mode: int = 1,
    dqn_train_episodes: int = 50,
    train_episodes: int = 200,
    eval_episodes: int = 20,
):
    """
    深度模型权重：
    - dqn_model_path: 普通 DQN（double_dqn=False）的 .pth；若提供且文件存在则不再训练该支路。
    - ddqn_model_path: DDQN / Ours（double_dqn=True）的 .pth；若提供且文件存在则不再训练该支路。
    未提供或文件不存在时，按 dqn_train_episodes 现场训练并保存为 dqn_model_*.pth / ddqn_model_*.pth。

    旧权重说明：
    - `DQN.py` 里 `run_training()` 保存的 `dqn_model_时间戳.pth` 与当前代码默认的 replay（Double DQN 目标）
      一致，评估时请作为 **DDQN/Ours** 使用：传 ddqn_model_path=该文件，且不要把它当 vanilla 用。
    - 对比脚本单独训练得到的 vanilla 一般为 `dqn_model_*.pth`（来自本脚本的 _train_and_save_dqn(False)）。
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _train_and_save_dqn(double_dqn: bool) -> Path:
        # 训练普通 DQN / Double DQN：环境与动作空间一致，差别只在 replay() target 公式
        dqn = DQN(mode="train", double_dqn=double_dqn)
        dqn.L = int(dqn_train_episodes)
        dqn.main(verbose=False)

        variant = "ddqn" if double_dqn else "dqn"
        model_path = PROJECT_ROOT / f"{variant}_model_{timestamp}.pth"
        dqn.save_model(model_path)
        return model_path

    def _eval_dqn(model_path: Path, double_dqn: bool):
        d = DQN(mode="test", double_dqn=double_dqn)
        d.L_test = eval_episodes
        d.load_model(str(model_path))

        mk_means, mk_stds = [], []
        u_means, u_stds = [], []
        tr_means, tr_stds = [], []
        for (m, e, j) in TEST_PARAMS:
            total_tr, total_u, total_mk, _ = d.main(fixed_params=(m, e, j), verbose=False)
            mk_means.append(float(np.mean(total_mk)))
            mk_stds.append(float(np.std(total_mk)))
            u_means.append(float(np.mean(total_u)))
            u_stds.append(float(np.std(total_u)))
            tr_means.append(float(np.mean(total_tr)))
            tr_stds.append(float(np.std(total_tr)))
        return mk_means, mk_stds, u_means, u_stds, tr_means, tr_stds

    # 1) 普通 DQN vs DDQN：优先使用已有 .pth，否则现场训练
    vanilla_ckpt = _resolve_checkpoint(dqn_model_path)
    ddqn_ckpt = _resolve_checkpoint(ddqn_model_path)

    if vanilla_ckpt is not None:
        dqn_model_path_trained = vanilla_ckpt
    else:
        dqn_model_path_trained = _train_and_save_dqn(double_dqn=False)

    if ddqn_ckpt is not None:
        ddqn_model_path_trained = ddqn_ckpt
    else:
        ddqn_model_path_trained = _train_and_save_dqn(double_dqn=True)

    (
        dqn_mk_means,
        dqn_mk_stds,
        dqn_u_means,
        dqn_u_stds,
        dqn_tr_means,
        dqn_tr_stds,
    ) = _eval_dqn(dqn_model_path_trained, double_dqn=False)

    (
        ddqn_mk_means,
        ddqn_mk_stds,
        ddqn_u_means,
        ddqn_u_stds,
        ddqn_tr_means,
        ddqn_tr_stds,
    ) = _eval_dqn(ddqn_model_path_trained, double_dqn=True)

    # 2) Q-learning
    q_cfg = TabularConfig(seed=20260329)
    q_agent = train_tabular_q_learning(train_episodes=train_episodes, reward_mode=reward_mode, q_config=q_cfg)

    mk_means, mk_stds, u_means, u_stds, tr_means, tr_stds = evaluate_tabular_agent_on_scenarios(
        agent=q_agent,
        reward_mode=reward_mode,
        eval_episodes=eval_episodes,
    )
    ql_mk_means, ql_mk_stds = mk_means, mk_stds
    ql_u_means, ql_u_stds = u_means, u_stds
    ql_tr_means, ql_tr_stds = tr_means, tr_stds

    # 3) SARSA
    s_cfg = TabularConfig(seed=20260330)
    s_agent = train_tabular_sarsa(train_episodes=train_episodes, reward_mode=reward_mode, q_config=s_cfg)

    mk_means, mk_stds, u_means, u_stds, tr_means, tr_stds = evaluate_tabular_agent_on_scenarios(
        agent=s_agent,
        reward_mode=reward_mode,
        eval_episodes=eval_episodes,
    )
    sarsa_mk_means, sarsa_mk_stds = mk_means, mk_stds
    sarsa_u_means, sarsa_u_stds = u_means, u_stds
    sarsa_tr_means, sarsa_tr_stds = tr_means, tr_stds

    # 4) 画图（与 test_runner 的风格一致：误差棒）
    scenario_labels = [_scenario_label(p) for p in TEST_PARAMS]
    x = np.arange(len(TEST_PARAMS))

    def _plot(metric_values, metric_stds, y_label, better_note, filename):
        plt.figure(figsize=(16, 8))
        plt.errorbar(
            x,
            metric_values["DDQN"],
            yerr=metric_stds["DDQN"],
            label="Ours",
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=3,
            alpha=0.95,
        )
        plt.errorbar(
            x,
            metric_values["DQN"],
            yerr=metric_stds["DQN"],
            label="DQN",
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=3,
            alpha=0.95,
        )
        plt.errorbar(x, metric_values["Q-learning"], yerr=metric_stds["Q-learning"], label="Q-learning",
                     marker="o", markersize=4, linewidth=1.5, capsize=3, alpha=0.95)
        plt.errorbar(x, metric_values["SARSA"], yerr=metric_stds["SARSA"], label="SARSA",
                     marker="o", markersize=4, linewidth=1.5, capsize=3, alpha=0.95)
        plt.xticks(x, scenario_labels, rotation=45, ha="right")
        plt.xlabel("Scenarios (n-m-e)")
        plt.ylabel(f"{y_label} ({better_note})")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        fig_path = PROJECT_ROOT / filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"图已保存: {fig_path}")

    _plot(
        metric_values={
            "DQN": dqn_mk_means,
            "DDQN": ddqn_mk_means,
            "Q-learning": ql_mk_means,
            "SARSA": sarsa_mk_means,
        },
        metric_stds={
            "DQN": dqn_mk_stds,
            "DDQN": ddqn_mk_stds,
            "Q-learning": ql_mk_stds,
            "SARSA": sarsa_mk_stds,
        },
        y_label="Average Makespan",
        better_note="lower is better",
        filename=f"compare_tabular_makespan_{timestamp}.png",
    )
    _plot(
        metric_values={
            "DQN": dqn_u_means,
            "DDQN": ddqn_u_means,
            "Q-learning": ql_u_means,
            "SARSA": sarsa_u_means,
        },
        metric_stds={
            "DQN": dqn_u_stds,
            "DDQN": ddqn_u_stds,
            "Q-learning": ql_u_stds,
            "SARSA": sarsa_u_stds,
        },
        y_label="Average Uave",
        better_note="higher is better",
        filename=f"compare_tabular_uave_{timestamp}.png",
    )
    _plot(
        metric_values={
            "DQN": dqn_tr_means,
            "DDQN": ddqn_tr_means,
            "Q-learning": ql_tr_means,
            "SARSA": sarsa_tr_means,
        },
        metric_stds={
            "DQN": dqn_tr_stds,
            "DDQN": ddqn_tr_stds,
            "Q-learning": ql_tr_stds,
            "SARSA": sarsa_tr_stds,
        },
        y_label="Average TRave",
        better_note="lower is better",
        filename=f"compare_tabular_trave_{timestamp}.png",
    )

    # 结果摘要不在控制台打印（避免输出过多）


if __name__ == "__main__":
    # reward_mode: 1=reward1(TR) 2=reward2(U) 3=Algorithm5 联合奖励（仅表格法训练用）
    # 使用已有权重示例（填文件名或绝对路径，存在则跳过对应 DQN/DDQN 训练）：
    #   dqn_model_path="dqn_model_20260330_120000.pth",
    #   ddqn_model_path="ddqn_model_20260330_120000.pth",
    # 旧版 DQN.py 训练得到的 dqn_model_*.pth 请作为 ddqn_model_path 传入（与 double_dqn=True 一致）。
    run_dqn_and_tabular_comparison(
        dqn_model_path=None,
        ddqn_model_path=None,
        reward_mode=3,
        dqn_train_episodes=300,
        train_episodes=100,
        eval_episodes=20,
    )