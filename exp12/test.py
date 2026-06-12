import torch
import math
import numpy as np

# ===================== 1. 实现 Sinusoidal 正弦位置编码 =====================
def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    标准 Transformer 正弦位置编码
    :param seq_len: 序列长度
    :param d_model: 词嵌入维度
    :return: pos_encoding: [seq_len, d_model]
    """
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    return pos_enc


# ===================== 2. 实现二维向量旋转 =====================
def rotate_2d(x: torch.Tensor, theta: float) -> torch.Tensor:
    """
    二维向量旋转
    :param x: 输入二维向量 [2]
    :param theta: 旋转角度(弧度)
    :return: 旋转后向量 [2]
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot_mat = torch.tensor([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ], dtype=x.dtype)
    return torch.matmul(rot_mat, x)


# ===================== 3. 实现高维 RoPE =====================
def rope_apply(x: torch.Tensor, seq_len: int, d_head: int) -> torch.Tensor:
    """
    对 Q/K 应用 RoPE 旋转位置编码
    :param x: 输入特征 [seq_len, d_head]
    :param seq_len: 序列长度
    :param d_head: 单头维度，必须为偶数
    :return: 旋转后特征 [seq_len, d_head]
    """
    assert d_head % 2 == 0, "RoPE 要求维度为偶数"
    x_rot = x.clone()

    for pos in range(seq_len):
        for i in range(0, d_head, 2):
            # 计算旋转角 theta
            theta = pos / (10000 ** (i / d_head))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # 取一对维度做旋转
            x1 = x[pos, i]
            x2 = x[pos, i + 1]
            x_rot[pos, i] = x1 * cos_t - x2 * sin_t
            x_rot[pos, i + 1] = x1 * sin_t + x2 * cos_t
    return x_rot


# ===================== 4. 对比 E+pos 与 RoPE 输入方式 =====================
def compare_e_pos_vs_rope():
    """对比 E+pos(加法嵌入) 和 RoPE(旋转嵌入) 两种位置注入方式"""
    seq_len = 8
    d_model = 16

    # 1. 随机词嵌入 E
    embed = torch.randn(seq_len, d_model)
    # 2. 正弦位置编码 pos
    pos_enc = sinusoidal_pos_encoding(seq_len, d_model)

    # -------- 方式1: E + pos 加法融合 --------
    e_plus_pos = embed + pos_enc
    print("=== E + pos (加法位置编码) ===")
    print(f"形状: {e_plus_pos.shape}, 数值样例:\n{e_plus_pos[:2, :4]}\n")

    # -------- 方式2: RoPE 旋转融合(作用在Q/K) --------
    rope_out = rope_apply(embed, seq_len, d_model)
    print("=== RoPE (旋转位置编码) ===")
    print(f"形状: {rope_out.shape}, 数值样例:\n{rope_out[:2, :4]}\n")


# ===================== 5. 数值实验：验证 RoPE 相对位置性质 =====================
def test_rope_relative_position():
    """
    正确验证 RoPE 的相对位置不变性：
    同一个 query 和 key，放在不同绝对位置但相同相对位置，点积结果一致
    """
    d_head = 4
    seq_len = 10

    # 固定 query 和 key 的原始值，避免随机初始化的干扰
    q = torch.tensor([1.0, 2.0, 3.0, 4.0])
    k = torch.tensor([5.0, 6.0, 7.0, 8.0])

    # 构造完整序列：所有位置都用同一个 q 和 k
    q_seq = q.repeat(seq_len, 1)
    k_seq = k.repeat(seq_len, 1)

    # 对整个序列应用 RoPE
    q_rope = rope_apply(q_seq, seq_len, d_head)
    k_rope = rope_apply(k_seq, seq_len, d_head)

    print("=== RoPE 相对位置验证实验 ===")
    print("验证：相同相对位置的 Q/K 点积近似相等")

    # 相对偏移 delta = 2
    delta = 2
    # 组1：位置 (1, 3) → 偏移 2
    score1 = torch.dot(q_rope[1], k_rope[1 + delta])
    # 组2：位置 (4, 6) → 偏移 2
    score2 = torch.dot(q_rope[4], k_rope[4 + delta])

    print(f"相对偏移 delta = {delta}")
    print(f"位置(1, 3) 注意力点积: {score1.item():.6f}")
    print(f"位置(4, 6) 注意力点积: {score2.item():.6f}")
    print("结论：两组绝对位置不同、相对位置相同，点积结果几乎一致，证明 RoPE 具备相对位置特性\n")


# ===================== 主函数：统一运行所有实验 =====================
if __name__ == "__main__":
    # 1. 测试正弦位置编码
    print("1. 正弦位置编码结果:")
    sin_pos = sinusoidal_pos_encoding(seq_len=5, d_model=8)
    print(sin_pos, "\n")

    # 2. 测试二维向量旋转
    print("2. 二维向量旋转测试:")
    vec_2d = torch.tensor([1.0, 0.0])
    rot_vec = rotate_2d(vec_2d, theta=math.pi/2)
    print(f"原向量: {vec_2d}, 旋转90度后: {rot_vec}\n")

    # 3. 对比 E+pos 与 RoPE
    compare_e_pos_vs_rope()

    # 4. RoPE 相对位置数值验证
    test_rope_relative_position()