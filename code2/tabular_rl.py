import numpy as np
from dataclasses import dataclass


@dataclass
class TabularConfig:
    num_states: int = 10
    num_actions: int = 9  # 9 条复合调度规则
    alpha: float = 0.1    # 学习率
    gamma: float = 0.95   # 折扣因子
    epsilon: float = 0.2  # ε-greedy 的探索率
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    seed: int = 20260329


class TabularQLearning:
    """
    Q-learning：更新目标对应公式（15）
    Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
    """

    def __init__(self, config: TabularConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.q = np.zeros((config.num_states, config.num_actions), dtype=np.float64)

    def select_action(self, state: int, *, epsilon: float | None = None) -> int:
        eps = self.cfg.epsilon if epsilon is None else float(epsilon)
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.cfg.num_actions))
        return int(np.argmax(self.q[state]))

    def decay_epsilon(self) -> None:
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        *,
        done: bool,
    ) -> None:
        current = self.q[s, a]
        if done:
            target = r
        else:
            target = r + self.cfg.gamma * float(np.max(self.q[s_next]))
        self.q[s, a] = current + self.cfg.alpha * (target - current)


class TabularSARSA:
    """
    SARSA：更新目标对应公式（16）
    Q(s,a) <- Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
    """

    def __init__(self, config: TabularConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.q = np.zeros((config.num_states, config.num_actions), dtype=np.float64)

    def select_action(self, state: int, *, epsilon: float | None = None) -> int:
        eps = self.cfg.epsilon if epsilon is None else float(epsilon)
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.cfg.num_actions))
        return int(np.argmax(self.q[state]))

    def decay_epsilon(self) -> None:
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        a_next: int,
        *,
        done: bool,
    ) -> None:
        current = self.q[s, a]
        if done:
            target = r
        else:
            target = r + self.cfg.gamma * float(self.q[s_next, a_next])
        self.q[s, a] = current + self.cfg.alpha * (target - current)

