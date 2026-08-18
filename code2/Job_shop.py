import numpy as np
import random
#from Instance_Generator import Processing_time,A,D,M_num,Op_num,J,O_num,J_num
from Object_for_FJSP import Object
from Maintain import PMMachine
from project_paths import PROJECT_ROOT

class Situation:
    def __init__(self,J_num,M_num,O_num,J,Processing_time,D,Ai,Change_cutter_time, Repair_time, EL):
        self.Ai=Ai                  #arriving time
        self.D=D                    #delivery time 交货期
        self.O_num=O_num            #operation num
        self.M_num=M_num            #machine num
        self.J_num=J_num            #job num
        self.J=J                    #operation num of each job
        self.Processing_time = Processing_time   # processing time
        self.CTK=[0 for i in range(M_num)]      #last machine working time
        self.OP=[0 for i in range(J_num)]       #the number of finished operations for each job
        self.UK=[0 for i in range(M_num)]       #machine utilization 机器利用率
        self.CRJ=[0 for i in range(J_num)]      #completion rate of jobs 工作完成率

        #初始化作业对象 创建 J_num 个 Object 对象，每个代表一个作业
        self.Jobs=[]
        for i in range(J_num):
            F = Object(i)
            self.Jobs.append(F)
        #初始化机器对象 创建 M_num 个 Object 对象，每个代表一台机器
        self.Machines = []
        for i in range(M_num):
            F = Object(i)
            self.Machines.append(F)

        #初始化机器维护对象
        self.PMMachines = []
        for i in range(M_num):
            F = PMMachine(i)
            self.PMMachines.append(F)


        self.Change_cutter_time=Change_cutter_time #换刀时间
        self.Repair_time=Repair_time #维修时间
        self.EL = EL #紧急程度
        # ---------------breakdown probability----------
        self.BP = 0.1 #故障概率
        # 当前事件驱动调度时刻：总是在“下一个可派工时刻”
        self.t_now = 0.0

    def _machine_ready_time(self, machine_idx: int) -> float:
        """机器下一次可用时刻（考虑维修/换刀状态）"""
        pm = self.PMMachines[machine_idx]
        pm.update_status(self.t_now)
        return max(float(self.CTK[machine_idx]), float(pm.down_until))

    def _advance_time_to_next_event(self):
        """
        事件驱动：当没有已到达的作业可派工时，推进到下一个作业到达时刻；
        同时也可能推进到机器从维修/换刀恢复的时刻（取两者的最小可推进点）。
        """
        unfinished_jobs = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if not unfinished_jobs:
            return

        # 下一个作业到达时刻
        next_arrival = min(float(self.Ai[j]) for j in unfinished_jobs if float(self.Ai[j]) > self.t_now) if any(float(self.Ai[j]) > self.t_now for j in unfinished_jobs) else None
        # 下一个机器恢复时刻
        next_machine_up = min(float(pm.down_until) for pm in self.PMMachines if float(pm.down_until) > self.t_now) if any(float(pm.down_until) > self.t_now for pm in self.PMMachines) else None

        candidates = [x for x in [next_arrival, next_machine_up] if x is not None]
        if candidates:
            self.t_now = min(candidates)

    def _Update(self,Job,Machine):
        self.CTK[Machine]=max(self.Machines[Machine].End) #更新机器的最新工作时间
        self.OP[Job]+=1 #更新作业的已完成操作数
        self.UK[Machine]=sum(self.Machines[Machine].T)/self.CTK[Machine] #计算机器利用率 累计加工时间/最新加工时间
        self.CRJ[Job]=self.OP[Job]/self.J[Job] #计算作业完成率
        # 事件驱动：更新到下一个“可派工”时刻（任意机器可用的最早时刻）
        self.t_now = min(self._machine_ready_time(k) for k in range(self.M_num))

    def Features(self): # 计算6个论文口径状态特征
        """
        论文的 6 个状态特征为：
        (Uave, Ustd, CRJave, CRJstd, TRave, TRstd)
        """
        # 1-2) Uave / Ustd
        U_ave = sum(self.UK) / self.M_num
        K = 0.0
        for uk in self.UK:
            K += np.square(uk - U_ave)
        U_std = np.sqrt(K / self.M_num)

        # 3-4) CRJave / CRJstd，其中 CRJi = OPTi / (OPTi + ETLi)
        # 5-6) TRave / TRstd，其中 TRi = (OPTi + ETLi - DDLi) / (OPTi + ETLi)
        CRJ_list = []
        TR_list = []
        for i in range(self.J_num):
            OPT_i = float(sum(self.Jobs[i].T))  # 已完成加工时间

            # 估计剩余加工时间 ETLi：对剩余每个工序求可选机器加工时间的平均值，再求和
            ETL_i = 0.0
            for op_idx in range(self.OP[i], self.J[i]):
                pt_list = [pt for pt in self.Processing_time[i][op_idx] if pt != -1 and pt > 0]
                if pt_list:
                    ETL_i += float(sum(pt_list)) / float(len(pt_list))

            denom = OPT_i + ETL_i
            if denom > 0:
                CRJ_i = OPT_i / denom
                DDL_i = float(self.D[i] - self.Ai[i])
                # 论文口径：TR_i = max(0, (OPT_i + ETL_i - DDL_i) / (OPT_i + ETL_i))
                TR_i = max(0.0, (denom - DDL_i) / denom)
            else:
                CRJ_i = 0.0
                TR_i = 0.0

            # 论文说明状态特征使用比率并映射到[0,1]以增强稳定性
            CRJ_list.append(float(np.clip(CRJ_i, 0.0, 1.0)))
            TR_list.append(float(np.clip(TR_i, 0.0, 1.0)))

        CRJ_ave = float(sum(CRJ_list)) / float(self.J_num)
        K = 0.0
        for x in CRJ_list:
            K += np.square(x - CRJ_ave)
        CRJ_std = np.sqrt(K / self.J_num)

        TR_ave = float(sum(TR_list)) / float(self.J_num)
        K = 0.0
        for x in TR_list:
            K += np.square(x - TR_ave)
        TR_std = np.sqrt(K / self.J_num)

        return U_ave, U_std, CRJ_ave, CRJ_std, TR_ave, TR_std

    #Composite dispatching rule 1
    #return Job,Machine
    def rule1(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = UC_Job[np.argmin([(self.OP[i]) / (self.J[i]) for i in UC_Job])] #选择尚未完成的作业中，已完成操作数最少的作业
        try: #计算该作业的结束时间 C_ij
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End) #选择该作业的下一个操作（On），On 是作业 Job_i 的下一个操作的索引

        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1: #检查该作业在当前操作（On）上是否有有效的加工时间
                PT = self.Processing_time[Job_i][On][i] * self.PMMachines[i].De_factor
                # tool change 如果有工具更换，则在加工时间上增加工具更换时间
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                # 过滤：故障/换刀状态机器在当前时刻不可用
                if not self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(9999)
                else:
                    Mk.append(max(C_ij, A_ij, self.CTK[i], self.PMMachines[i].down_until) + PT)
            else:
                Mk.append(9999) #如果该机器无法处理该作业，则将该机器的预计开始时间设置为9999
        # print('This is from rule 1:',Mk)
        Machine=np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i,Machine

    # Composite dispatching rule 2
    # 与规则1相似，但是去除了更换工具的部分
    # return Job,Machine
    def rule2(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = UC_Job[np.argmin([(self.OP[i]) / (self.J[i]) for i in UC_Job])]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)

        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                if not self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(9999)
                else:
                    Mk.append(max(C_ij, A_ij, self.CTK[i], self.PMMachines[i].down_until))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3
    # 与规则1相似，但是选择机器时采用了随机选择
    # return Job,Machine
    def rule3(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = UC_Job[np.argmin([(self.OP[i]) / (self.J[i]) for i in UC_Job])]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                if self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(i)
        # print('This is from rule 1:',Mk)
        if not Mk:
            Mk = [i for i in range(len(self.CTK)) if self.Processing_time[Job_i][On][i] != -1]
        Machine = random.choice(Mk)             #随机选了一台机器
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4
    # 根据当前系统的平均结束时间计算延迟作业（Tard_Job）和未完成作业（UC_Job）选择工作，后半部分与规则1相同
    # return Job,Machine
    def rule4(self):
        T_cur = self.t_now

        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and float(self.Ai[i]) <= self.t_now and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            T_cur = self.t_now
            Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and float(self.Ai[i]) <= self.t_now and self.D[i] < T_cur]
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        #根据作业的延迟情况选择作业
        if Tard_Job == []: #无超期作业时，根据紧急程度等判断
            jobs = []
            for i in UC_Job:
                try:
                    C_i = max(self.Jobs[i].End)
                except:
                    C_i = self.Ai[i]
                jobs.append((C_i + T_cur - self.D[i]) / self.EL[i]) #表示作业的延迟程度和其优先级
            Job_i = UC_Job[np.argmin(jobs)]
        else: #有超期作业时，则根据作业的延迟情况选择超期作业中的一个
            T_ijave = []
            for i in Tard_Job:
                T_ijave.append(self.D[i] - T_cur / (3- self.EL[i] + 1))
            Job_i = Tard_Job[np.argmin(T_ijave)]

        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i] * self.PMMachines[i].De_factor
                # tool change
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                if not self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(9999)
                else:
                    Mk.append(max(C_ij, A_ij, self.CTK[i], self.PMMachines[i].down_until) + PT)
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5
    # 与规则4相似，但是去除了更换工具的部分
    # return Job,Machine
    def rule5(self):
        T_cur = self.t_now
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and float(self.Ai[i]) <= self.t_now and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            T_cur = self.t_now
            Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and float(self.Ai[i]) <= self.t_now and self.D[i] < T_cur]
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            jobs = []
            for i in UC_Job:
                try:
                    C_i = max(self.Jobs[i].End)
                except:
                    C_i = self.Ai[i]
                jobs.append((C_i + T_cur - self.D[i]) / self.EL[i])
            Job_i = UC_Job[np.argmin(jobs)]
        else:
            T_ijave = []
            for i in Tard_Job:
                T_ijave.append(self.D[i] - T_cur / (3 - self.EL[i] + 1))
            Job_i = Tard_Job[np.argmin(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                if not self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(9999)
                else:
                    Mk.append(max(C_ij, A_ij, self.CTK[i], self.PMMachines[i].down_until))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6
    # 与规则4相似，但是选择机器时采用了随机选择
    # return Job,Machine
    def rule6(self):
        T_cur = self.t_now
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and float(self.Ai[i]) <= self.t_now and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            T_cur = self.t_now
            Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and float(self.Ai[i]) <= self.t_now and self.D[i] < T_cur]
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            jobs = []
            for i in UC_Job:
                try:
                    C_i = max(self.Jobs[i].End)
                except:
                    C_i = self.Ai[i]
                jobs.append((C_i + T_cur - self.D[i]) / self.EL[i])
            Job_i = UC_Job[np.argmin(jobs)]
        else:
            T_ijave = []
            for i in Tard_Job:
                T_ijave.append(self.D[i] - T_cur / (3 - self.EL[i] + 1))
            Job_i = Tard_Job[np.argmin(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                if self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(i)
        # print('This is from rule 1:',Mk)
        if not Mk:
            Mk = [i for i in range(len(self.CTK)) if self.Processing_time[Job_i][On][i] != -1]
        Machine = random.choice(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7
    # 随机选择工作，后半部分与规则1相同
    # return Job,Machine
    def rule7(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now] #获取已到达且未完成的作业
        if UC_Job == []:
            self._advance_time_to_next_event()
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job) #随机选择一个未完成的作业
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i] * self.PMMachines[i].De_factor
                # tool change
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                if not self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(9999)
                else:
                    Mk.append(max(C_ij, A_ij, self.CTK[i], self.PMMachines[i].down_until) + PT)
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 8
    # 与规则7相似，但是去除了更换工具的部分
    # return Job,Machine
    def rule8(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job)
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                if not self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(9999)
                else:
                    Mk.append(max(C_ij, A_ij, self.CTK[i], self.PMMachines[i].down_until))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 9
    # 与规则7相似，但是选择机器时采用了随机选择
    # return Job,Machine
    def rule9(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            self._advance_time_to_next_event()
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j] and float(self.Ai[j]) <= self.t_now]
        if UC_Job == []:
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job)
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                if self.PMMachines[i].is_available_at(self.t_now):
                    Mk.append(i)
        # print('This is from rule 1:',Mk)
        if not Mk:
            Mk = [i for i in range(len(self.CTK)) if self.Processing_time[Job_i][On][i] != -1]
        Machine = random.choice(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine


    #根据确定的混合调度规则执行调度
    def scheduling(self,action):
        Job,Machine=action[0],action[1]  #获取工作和所分配的机器
        O_n=len(self.Jobs[Job].End)      #该作业已完成的操作数量
        # print(Job, Machine,O_n)
        Idle=self.Machines[Machine].idle_time() #获取机器的空闲时间
        try:
            last_ot=max(self.Jobs[Job].End) #last_ot 是作业 Job 的上一个操作的结束时间。如果作业没有完成任何操作，则设置为0
        except:
            last_ot=0
        try:
            last_mt=max(self.Machines[Machine].End) #last_mt 是机器 Machine 的上一个操作的结束时间。如果机器没有进行过操作，则设置为0
        except:
            last_mt=0

        pm = self.PMMachines[Machine]

        # 事件驱动：当前时刻下，作业未到达/机器处于维修或换刀时均不可开工
        earliest_start = max(last_ot, last_mt, float(self.Ai[Job]), float(pm.down_until))
        Start_time = earliest_start

        # 换刀：将机器置为 tool-changing 状态一段时间（机器不可用），再开始加工
        if self.change_cutter(Job, Machine) == 1:
            pm.block_for(Start_time, self.Change_cutter_time[Machine], status="tool-changing")
            Start_time = max(Start_time, float(pm.down_until))

        # 预防性维护：若需要，先进入 repair 状态（不可用），再开始加工
        if pm.needs_pm():
            file = open(PROJECT_ROOT / 'PM.txt', 'a', encoding='utf-8')
            file.write(f"时间 {Start_time:.1f}: 机器 {Machine} 进行预防性维护, 健康指数={pm.health:.2f}, 故障概率={pm.failure_prob:.3f}")
            file.write("\n")
            file.flush()
            file.close()

            pm.block_for(Start_time, self.Repair_time[Machine] / 2, status="repair")
            # PM 后恢复
            pm.health = 100
            pm.usage_time = 0
            pm.failure_prob = 0.0
            Start_time = max(Start_time, float(pm.down_until))

        # 故障：在开工前发生，则进入 repair 状态（不可用），修复后再开工
        break_down = random.random()
        uk_bp = np.percentile(self.CTK, 90)
        if self.CTK[Machine] >= uk_bp:
            break_down = min(break_down, random.random())
        if break_down < pm.failure_prob:
            file = open(PROJECT_ROOT / 'PM.txt', 'a', encoding='utf-8')
            file.write(f"时间 {Start_time:.1f}: 机器 {Machine} 进行故障性维护")
            file.write("\n")
            file.flush()
            file.close()

            pm.block_for(Start_time, self.Repair_time[Machine], status="repair")
            # 故障维修只能部分恢复
            pm.health = round(min(90, pm.health + random.uniform(20, 40)), 2)
            pm.usage_time = pm.usage_time * 0.5
            Start_time = max(Start_time, float(pm.down_until))

        # 加工时间（受退化影响）
        PT = self.Processing_time[Job][O_n][Machine]
        pm.De_factor = 1 + round(0.01 * np.exp(0.05 * (100 - round(pm.health, 1))), 2)
        PT = PT * pm.De_factor
        print(f"此时退化指数为 {pm.De_factor}")


        for i in range(len(Idle)): #判断空闲时间段是否可以用于处理当前作业
            if Idle[i][1]-Idle[i][0]>PT: #如果存在空闲时间段且长度大于处理时间 PT，则选择合适的开始时间
                # 前插时也必须满足 earliest_start，避免绕过作业到达/机器不可用约束
                if Idle[i][0]>earliest_start: #如果空闲时间段开始晚于最早可开工时刻，则在该段开始
                    Start_time=Idle[i][0]
                if Idle[i][0]<earliest_start and Idle[i][1]-earliest_start>PT:
                    Start_time=earliest_start

        # 双重保护：开始时间不早于事件约束下界
        Start_time = max(Start_time, earliest_start)

        end_time=Start_time+PT #计算作业的结束时间 即开始时间+处理时间
        self.Machines[Machine]._add(Start_time,end_time,Job,PT) #添加机器操作记录
        self.Jobs[Job]._add(Start_time,end_time,Machine,PT)     #添加工作操作记录
        # 工序完成后更新健康（仅按加工耗时计，不把换刀/维修计入磨损）
        pm.update_health(self.Processing_time[Job][O_n][Machine] * pm.De_factor)
        self._Update(Job,Machine) #更新调度状态

    #平均延迟率主导的奖励函数
    def reward1(self, TR_t, TR_t1):
        """
        对齐论文：Reward algorithm 2（基于 TRave）。
        rt = +1  如果 TRave(t+1) < TRave(t)
        rt =  0  如果 TRave(t+1) < TRave(t) * 1.1
        rt = -1  否则
        """
        TR_t = float(TR_t)
        TR_t1 = float(TR_t1)

        # 由于状态特征已裁剪到[0,1]，此处无需额外的异常处理
        if TR_t1 < TR_t:
            return 1
        if TR_t1 < TR_t * 1.1:
            return 0
        return -1

    #平均机器利用率主导的奖励函数
    def reward2(self,U_t,U_t1):
        '''
               :param U_t: U_ave(t) 当前时刻 t 的机器平均利用率（U_ave(t)）
               :param U_t1: U_ave(t+1) 下一个时刻 t+1 的机器平均利用率（U_ave(t+1)）
               :return: reward
        '''
        if U_t1 > U_t: #下一个时刻的机器平均利用率大于当前时刻，利用率增加，奖励为 +1
            rt = 1
        else:
            if U_t1 > 0.9 * U_t: #利用率略微下降，奖励为 0
                rt = 0
            else: #利用率显著下降，奖励为 -1
                rt = -1
        return rt

    def reward3(self, TR_t, TR_t1, U_t, U_t1):
        """
        Algorithm 5 / reward algorithm 3：同时依据 TRave 与 Uave 的变化给奖励。
        TR_t, TR_t1: 时刻 t 与 t+1 的 TR_ave（状态特征第 5 维）
        U_t, U_t1: 时刻 t 与 t+1 的 U_ave（状态特征第 1 维）
        """
        TR_t, TR_t1 = float(TR_t), float(TR_t1)
        U_t, U_t1 = float(U_t), float(U_t1)
        if TR_t1 < TR_t:
            return 1
        if TR_t1 < TR_t * 1.1:
            return 0
        if U_t1 > U_t:
            return 1
        if U_t1 > U_t * 0.9:
            return 0
        return -1

    # tool change 判断是否需要更换刀具
    def change_cutter(self,Job,Machine):
        assigned_jobs = self.Machines[Machine].assign_for #表示当前机器（Machine）已经分配的作业列表。它是该机器分配给的作业
        assigned_machines = self.Jobs[Job].assign_for     #表示当前作业（Job）已经分配的机器列表。它是该作业分配给的机器
        #检查作业是否更换机器：如果当前作业 (Job) 已经分配了机器，并且最后分配的机器不是当前机器 Machine，则说明该作业需要在不同的机器上执行，因此需要更换刀具
        #检查机器是否更换作业：如果当前机器 (Machine) 已经分配了作业，并且最后分配的作业不是当前作业 Job，则说明该机器需要处理不同的作业，因此也需要更换刀具。
        #如果上述两条中满足任意一条，则需要更换刀具
        if (len(assigned_machines) != 0 and assigned_machines[-1] != Machine) or (len(assigned_jobs) != 0 and assigned_jobs[-1] != Job):
            return 1
        return 0
#Sit=Situation(J_num,M_num,O_num,J,Processing_time,D,A)