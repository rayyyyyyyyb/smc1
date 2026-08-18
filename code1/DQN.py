import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Job_shop import Situation
import time
from datetime import datetime
from pathlib import Path

from project_paths import PROJECT_ROOT

# 设置GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DQNNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # 隐藏层
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class DQN:
    def __init__(self, mode='train'):
        self.Hid_Size1 = 10
        self.Hid_Size2 = 50
        self.mode = mode  # 'train' 或 'test'

        # 网络结构
        self.model = DQNNetwork(6, [self.Hid_Size1, self.Hid_Size1, self.Hid_Size1], 2).to(device)
        self.model1 = DQNNetwork(7, [self.Hid_Size2] * 7, 9).to(device)

        if mode == 'train':
            # 训练模式：初始化优化器
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.model1_optimizer = optim.Adam(self.model1.parameters(), lr=0.001)

        # Q-network 参数
        self.gama = 0.95
        self.global_step = 0
        self.update_target_steps = 200

        # 目标网络
        self.target_model = DQNNetwork(6, [self.Hid_Size1, self.Hid_Size1, self.Hid_Size1], 2).to(device)
        self.target_model1 = DQNNetwork(7, [self.Hid_Size2] * 7, 9).to(device)
        self.replace_target()

        # Agent 参数
        self.e_greedy = 0.6 if mode == 'train' else 0.0  # 测试时不探索
        self.e_greedy_decrement = 0.0001 if mode == 'train' else 0.0  # 测试时不递减
        self.L = 200  # 训练轮数
        self.L_test = 20  # 测试轮数

        # 经验回放缓冲区（只在训练时使用）
        if mode == 'train':
            self.buffer = deque(maxlen=2000)
            self.Batch_size = 16

    def replace_target(self):
        """将当前模型权重复制到目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model1.load_state_dict(self.model1.state_dict())

    def replay(self):
        """经验回放（只在训练时使用）"""
        if self.mode != 'train':
            return

        if self.global_step % self.update_target_steps == 0:
            self.replace_target()

        if len(self.buffer) < self.Batch_size:
            return

        # 从经验回放池中随机采样
        minibatch = random.sample(self.buffer, self.Batch_size)

        # 准备批量数据
        states = []
        actions = []
        rewards = []
        next_states = []
        reward_ids = []
        dones = []

        for state, action, reward, next_state, reward_id, done in minibatch:
            states.append(state[0])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state[0])
            reward_ids.append(reward_id)
            dones.append(done)

        # 转换为PyTorch张量
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        reward_ids = torch.LongTensor(reward_ids).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # 训练第一个网络 (model)
        self.model_optimizer.zero_grad()
        current_q_values = self.model(states)
        current_q = current_q_values.gather(1, reward_ids.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_online = self.model(next_states)
            best_actions = torch.argmax(next_q_values_online, dim=1)
            next_q_values_target = self.target_model(next_states)
            next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gama * next_q * (~dones).float()

        loss1 = F.mse_loss(current_q, target_q)
        loss1.backward()
        self.model_optimizer.step()

        # 训练第二个网络 (model1)
        self.model1_optimizer.zero_grad()

        with torch.no_grad():
            model_output = self.model(states)
            k_values = torch.max(model_output, dim=1)[0]
            states1 = torch.cat([states, k_values.unsqueeze(1)], dim=1)

            next_model_output = self.model(next_states)
            next_k_values = torch.max(next_model_output, dim=1)[0]
            next_states1 = torch.cat([next_states, next_k_values.unsqueeze(1)], dim=1)

        current_q_values1 = self.model1(states1)
        current_q1 = current_q_values1.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_online1 = self.model1(next_states1)
            best_actions1 = torch.argmax(next_q_values_online1, dim=1)
            next_q_values_target1 = self.target_model1(next_states1)
            next_q1 = next_q_values_target1.gather(1, best_actions1.unsqueeze(1)).squeeze(1)
            target_q1 = rewards + self.gama * next_q1 * (~dones).float()

        loss2 = F.mse_loss(current_q1, target_q1)
        loss2.backward()
        self.model1_optimizer.step()

        self.global_step += 1

    def Select_action(self, obs):
        """选择动作（测试时不探索）"""
        # obs 约定为 shape=(1, 6) 的数组
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

        if random.random() < self.e_greedy:  # 训练时探索
            rt = random.randint(0, 1)
            act = random.randint(0, 8)
        else:  # 利用学到的策略
            with torch.no_grad():
                output = self.model(obs_tensor)
                rt = torch.argmax(output).item()
                # model1 的输入为 7 维：6 个特征 + 1 个由 model 输出得到的 k 值
                # 这里使用“最大 Q 值”而不是“最大 Q 的索引”，与 replay() 里 k_values 的口径一致
                k_value = torch.max(output, dim=1, keepdim=True)[0]  # shape=(batch=1, 1)
                input_tensor = torch.cat([obs_tensor, k_value], dim=1)  # shape=(1, 7)
                act = torch.argmax(self.model1(input_tensor), dim=1).item()

        if self.mode == 'train':
            self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_decrement)

        return act, rt

    def _append(self, exp):
        """将经验存入缓冲区（只在训练时使用）"""
        if self.mode == 'train':
            self.buffer.append(exp)

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model1_state_dict': self.model1.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'target_model1_state_dict': self.target_model1.state_dict(),
            'global_step': self.global_step,
            'e_greedy': self.e_greedy
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            # 新版 PyTorch 建议显式设置 weights_only=True，降低反序列化风险；
            # 兼容旧版（不支持该参数）时自动回退。
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model1.load_state_dict(checkpoint['model1_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.target_model1.load_state_dict(checkpoint['target_model1_state_dict'])
            self.global_step = checkpoint['global_step']
            self.e_greedy = checkpoint['e_greedy']
            print(f"模型已从 {path} 加载")
        else:
            print(f"警告: 模型文件 {path} 不存在，使用初始模型")

    def Instance_Generator(self, M_num=None, E_ave=None, New_insert=None, mode='train'):
        """实例生成器。

        - 初始作业：固定 Initial_Job_num=5 件，到达时刻 Ai=0，交期 Di 仅按紧急度 EL 与加工量估计（无到达偏移）。
        - 插入作业：共 New_insert 件，到达间隔 ~Exp(E_ave) 累加得到 Ai>0，交期 Di = Ai + (0.2+0.5*EL)*T_ijave。
        总作业数 n = Initial_Job_num + New_insert；TR_ave 对全部 n 件作业的 TR_i 取算术平均，不区分类型。
        """
        if mode == 'train':
            # 训练模式：使用随机范围
            if M_num is None:
                M_num = random.randint(8, 18)
            if E_ave is None:
                E_ave = random.randint(50, 200)
            if New_insert is None:
                New_insert = random.randint(10, 30)
        else:
            # 测试模式：使用固定值
            if M_num is None:
                M_num = 8
            if E_ave is None:
                E_ave = 50
            if New_insert is None:
                New_insert = 10

        Initial_Job_num = 5  # 初始作业数量（与插入作业共同构成 J_num）
        Op_num = [random.randint(1, 20) for i in range(New_insert + Initial_Job_num)]

        Processing_time = []
        for i in range(Initial_Job_num + New_insert):
            Job_i = []
            for j in range(Op_num[i]):
                k = random.randint(1, M_num - 2)
                T = list(range(M_num))
                random.shuffle(T)
                T = T[0:k + 1]
                O_i = list(np.ones(M_num) * (-1))
                for M_i in range(len(O_i)):
                    if M_i in T:
                        O_i[M_i] = random.randint(1, 50)
                Job_i.append(O_i)
            Processing_time.append(Job_i)

        A1 = [0 for i in range(Initial_Job_num)]
        # 插入作业到达：泊松过程 => 间隔服从指数分布 Exp(E_ave)，累加得到各插入作业的到达时刻 Ai
        intervals = np.random.exponential(E_ave, size=New_insert)
        intervals = [int(x) for x in intervals]
        A = list(np.cumsum(intervals).astype(int))
        A1.extend(A)

        EL = [random.randint(1, 3) for i in range(len(A1))]

        T_ijave = []
        for i in range(Initial_Job_num + New_insert):
            Tad = []
            for j in range(Op_num[i]):
                T_ijk = [k for k in Processing_time[i][j] if k != -1]
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(sum(Tad))
        D1 = [int((0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in range(Initial_Job_num)]
        D = [int(A1[i] + (0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in
             range(Initial_Job_num, Initial_Job_num + New_insert)]
        D1.extend(D)

        O_num = sum(Op_num)
        J = dict(enumerate(Op_num))
        J_num = Initial_Job_num + New_insert

        Change_cutter_time = list(np.zeros(M_num))
        Repair_time = list(np.zeros(M_num))
        for i in range(M_num):
            Change_cutter_time[i] = random.randint(1, 50)
            Repair_time[i] = random.randint(1, 99)

        return Processing_time, A1, D1, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL

    def main(self, fixed_params=None, verbose=True, enable_preventive_maintenance=True):
        """主运行函数"""
        k = 0
        Total_tard = []
        Total_makespan = []
        Total_uk_ave = []
        TR = []

        # 确定循环次数
        num_episodes = self.L if self.mode == 'train' else self.L_test

        for episode_idx in range(num_episodes):
            Total_reward = 0
            if verbose:
                print(f'-----------------------{self.mode} {episode_idx + 1} ------------------------------')

            # 生成实例
            if self.mode == 'train':
                Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = self.Instance_Generator(
                    mode='train')
            else:
                M_num_fixed, E_ave_fixed, New_insert_fixed = fixed_params
                Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = self.Instance_Generator(
                    M_num_fixed, E_ave_fixed, New_insert_fixed, mode='test')

            obs = [0 for _ in range(6)]
            obs = np.expand_dims(obs, 0)
            done = False

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
                enable_preventive_maintenance=enable_preventive_maintenance,
            )

            for op_idx in range(O_num):
                k += 1
                at, rt = self.Select_action(obs)

                # 执行调度规则（由 DQN 在 9 条复合规则中选一条）
                rule_mapping = {
                    0: Sit.rule1, 1: Sit.rule2, 2: Sit.rule3,
                    3: Sit.rule4, 4: Sit.rule5, 5: Sit.rule6,
                    6: Sit.rule7, 7: Sit.rule8, 8: Sit.rule9
                }
                at_trans = rule_mapping[at]()

                if verbose:
                    print('The', op_idx, 'th operation>>', 'select action:', at, ' ', 'job ', at_trans[0],
                          'is assigned for machine ', at_trans[1])
                Sit.scheduling(at_trans)
                obs_t = Sit.Features()

                if op_idx == O_num - 1:
                    done = True

                obs_t = np.expand_dims(obs_t, 0)

                if 0 == rt:
                    # reward1 对齐论文：基于 TRave(t) -> reward algorithm 2
                    r_t = Sit.reward1(obs[0][4], obs_t[0][4])
                else:
                    # reward2 对齐论文：基于 Uave(t) -> reward algorithm 1
                    r_t = Sit.reward2(obs[0][0], obs_t[0][0])

                self._append((obs, at, r_t, obs_t, rt, done))

                # 只在训练时进行经验回放
                if self.mode == 'train' and k > self.Batch_size:
                    self.replay()

                Total_reward += r_t
                obs = obs_t

            # 计算指标
            total_TR = 0
            makespan = 0
            uk_ave = sum(Sit.UK) / M_num
            Job = Sit.Jobs

            for Ji in range(len(Job)):
                endTime = max(Job[Ji].End)
                makespan = max(makespan, endTime)
                # TR_i = (OPT_i + ETL_i - DDL_i) / (OPT_i + ETL_i)
                OPT_i = sum(Job[Ji].T)
                ETL_i = 0
                for op_idx in range(Sit.OP[Ji], Sit.J[Ji]):
                    pt_list = [pt for pt in Sit.Processing_time[Ji][op_idx] if pt != -1 and pt > 0]
                    if pt_list:
                        ETL_i += sum(pt_list) / len(pt_list)

                denom = OPT_i + ETL_i
                if denom > 0:
                    # 由于 D[Ji] 是绝对到期时刻，而 OPT/ETL 是“加工时间”口径，
                    # 这里使用 D[Ji] - A[Ji] 将到期时间转换为相对到达后的加工时间口径。
                    DDL_i = D[Ji] - A[Ji]
                    # 论文口径：无延迟则记为 0
                    TR_i = max(0.0, (denom - DDL_i) / denom)
                else:
                    TR_i = 0

                total_TR += TR_i

            # TR_ave = (Σ_i TR_i) / n；n 含初始作业与插入作业，单件 TR_i 定义同 Features（见 Job_shop）
            TR_ave = total_TR / len(Job)
            TR_ave = round(TR_ave, 6)
            uk_ave = round(uk_ave, 4)
            makespan = round(makespan, 2)

            if verbose:
                print('<<<<<<<<<-----------------TR_ave:', TR_ave, '------------------->>>>>>>>>>')
            Total_tard.append(TR_ave)
            if verbose:
                print('<<<<<<<<<-----------------uk_ave:', uk_ave, '------------------->>>>>>>>>>')
            Total_uk_ave.append(uk_ave)
            if verbose:
                print('<<<<<<<<<-----------------makespan:', makespan, '------------------->>>>>>>>>>')
            Total_makespan.append(makespan)
            if verbose:
                print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)

        return Total_tard, Total_uk_ave, Total_makespan, TR

def run_training():
    """训练函数"""
    print("开始训练（随机参数范围）...")
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    d = DQN(mode='train')
    Total_tard, Total_uk_ave, Total_makespan, TR = d.main()

    # 保存模型
    model_path = PROJECT_ROOT / f'dqn_model_{timestamp}.pth'
    d.save_model(model_path)

    # 固定实例测试（训练结束后额外评估一次）
    fixed_params = (8, 50, 10)  # (M_num, E_ave, New_insert)
    fixed_seed = 20260327
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    d_test = DQN(mode='test')
    d_test.L_test = 20
    d_test.load_model(str(model_path))
    Total_tard_f, Total_uk_ave_f, Total_makespan_f, TR_f = d_test.main(fixed_params=fixed_params)
    print(
        "固定实例测试结果: "
        f"Machine={fixed_params[0]}, E_ave={fixed_params[1]}, Job_insert={fixed_params[2]}, "
        f"TR_ave={Total_tard_f[0]:.6f}, uk_ave={Total_uk_ave_f[0]:.6f}, makespan={Total_makespan_f[0]:.2f}, "
        f"reward={TR_f[0]:.6f}"
    )

    # 计算性能指标
    TR_ave = round(sum(Total_tard) / d.L, 2)
    uk_ave = round(sum(Total_uk_ave) / d.L, 4)
    makespan_ave = round(sum(Total_makespan) / d.L, 2)

    std1 = round(np.sqrt(sum(np.square(ta - TR_ave) for ta in Total_tard) / d.L), 2)
    std2 = round(np.sqrt(sum(np.square(ua - uk_ave) for ua in Total_uk_ave) / d.L), 2)
    std3 = round(np.sqrt(sum(np.square(ma - makespan_ave) for ma in Total_makespan) / d.L), 2)

    result_line = f"{TR_ave},{std1},{uk_ave},{std2},{makespan_ave},{std3}"
    print(f"训练结果: {result_line}")

    # 保存结果（追加模式）
    with open(PROJECT_ROOT / 'results_train.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n=== 训练结果 {timestamp} ===\n")
        file.write("训练模式: 随机参数范围\n")
        file.write(f"参数范围: 机器数(8-18), 平均到达时间(50-200), 新任务数(10-30)\n")
        file.write(
            "指标(episode均值±标准差): "
            f"TR_ave={TR_ave}, TR_std={std1}, "
            f"uk_ave={uk_ave}, uk_std={std2}, "
            f"makespan_ave={makespan_ave}, makespan_std={std3}\n"
        )
        file.write(f"TRave: {Total_tard}\n")
        file.write(f"机器利用率: {Total_uk_ave}\n")
        file.write(f"最大完成时间: {Total_makespan}\n")
        file.write(f"奖励: {TR}\n")
        file.write("=" * 50 + "\n")

    # 保存详细数据（追加模式）
    with open(PROJECT_ROOT / 'data_train.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n=== 训练详细数据 {timestamp} ===\n")
        file.write(f"TRave：{Total_tard}\n")
        file.write(f"uk：{Total_uk_ave}\n")
        file.write(f"makespan：{Total_makespan}\n")
        file.write(f"rewards：{TR}\n")
        file.write("=" * 50 + "\n")

    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f} 秒")

    return model_path


if __name__ == '__main__':
    # 仅执行训练；测试请运行独立脚本 test_runner.py
    run_training()