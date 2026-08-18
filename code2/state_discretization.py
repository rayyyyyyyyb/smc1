import numpy as np
from typing import Iterable


def discretize_state_from_features(
    ddqn_features: Iterable[float],
    *,
    num_states: int = 10,
    bin_size: float = 0.1,
) -> int:
    """
    按公平性要求构造离散状态 S(t)*：
    - 先取“DDQN 的状态特征分量”的平均值（这里用 ddqn_features 的平均值）
    - 然后用区间长度 0.1 分箱到 10 个状态：S1..S10

    规则（等价于你给的文字描述）：
    - 若 S(t)* ∈ [0, 0.1] -> 返回 0 (S1)
    - 若 S(t)* ∈ (0.1, 0.2] -> 返回 1 (S2)
    - ...依此类推直到 (0.9, 1.0] -> 返回 9 (S10)
    """
    features = list(ddqn_features)
    if not features:
        # 没有特征时退化为第一个离散状态
        return 0

    s_star = float(np.mean(features))
    # 你的特征通常已裁剪到[0,1]，这里做数值保护
    s_star = float(np.clip(s_star, 0.0, 1.0))

    # 用“<= 上边界”实现 [0,0.1] 属于S1 的边界约定
    for state_idx in range(num_states):
        upper = (state_idx + 1) * bin_size if state_idx < num_states - 1 else 1.0
        if s_star <= upper:
            return state_idx

    return num_states - 1

