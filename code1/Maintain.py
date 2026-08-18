import numpy as np
import random


class PMMachine:
    def __init__(self, id):
        self.id = id
        self.health = 100  # 初始健康状态 100
        self.usage_time = 0  # 运行时间
        self.repair_time = random.randint(1, 99)  # 初始维修时间

        # 机器状态：normal / tool-changing / repair
        self.status = "normal"
        # 机器由于换刀/维修等原因不可用直到该时刻
        self.down_until = 0.0

        # Weibull 分布参数
        self.shape = 2  # 形状参数 k
        self.scale = 500  # 尺度参数 λ
        self.failure_prob = 0.0  # 初始故障概率

        # 退化效应系数
        self.De_factor = 1

    def update_health(self, PT):
        """模拟设备运行，降低健康指数"""
        self.usage_time += PT
        wear = random.uniform(4, 8)  # 随机磨损
        self.health = round(max(0, self.health - wear), 2)  # 健康值下降

        # 计算故障概率（Weibull CDF）
        self.failure_prob = round(1 - np.exp(- (self.usage_time / self.scale) ** self.shape), 3)

        print(f"机器 {self.id} ：健康指数={self.health:.2f}，故障概率={self.failure_prob:.3f}")

    def update_status(self, t_now: float):
        """根据当前时刻更新机器状态"""
        if self.status != "normal" and t_now >= self.down_until:
            self.status = "normal"

    def is_available_at(self, t_now: float) -> bool:
        """在 t_now 时刻是否可用（不在换刀/维修状态）"""
        self.update_status(t_now)
        return self.status == "normal"

    def block_for(self, t_start: float, duration: float, status: str):
        """
        将机器置为不可用一段时间（换刀/维修）。
        不可用区间：[max(down_until, t_start), max(down_until, t_start)+duration]
        """
        t0 = max(float(self.down_until), float(t_start))
        self.down_until = t0 + float(duration)
        self.status = status

    def needs_pm(self):
        """判断是否需要维修"""
        # return self.health < 30 or random.random() < self.failure_prob  # 低于 30 或基于威布尔概率
        return self.failure_prob > 0.2 or self.health < 30  # 基于威布尔概率或低于 30