"""
SwiGLU 前馈网络实现

SwiGLU (Swish-Gated Linear Unit) 是 LLaMA 等现代 LLM 中使用的 FFN 变体。
公式: SwiGLU(x) = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T

其中:
- SiLU (Swish) = x * sigmoid(x)
- GLU (Gated Linear Unit) 使用门控机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def swiglu_ffn(d_model: int, d_ff: int | None = None) -> int:
    """
    计算 SwiGLU 的内部维度 d_ff。

    根据 LLaMA 论文，d_ff 应该约为 (8/3) * d_model，
    并且需要是 64 的倍数以优化硬件利用率。

    Args:
        d_model: 模型隐藏层维度
        d_ff: 可选的显式指定 d_ff，如果提供则直接返回

    Returns:
        计算后的 d_ff 值，保证是 64 的倍数
    """
    if d_ff is not None:
        return d_ff

    # 计算 (8/3) * d_model 并向上取整到最近的 64 的倍数
    target_d_ff = int(8 * d_model / 3)
    # 向上取整到 64 的倍数
    d_ff = ((target_d_ff + 63) // 64) * 64
    return d_ff


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络

    SwiGLU = Swish (SiLU) + GLU (Gated Linear Unit)

    结构:
        1. w1: 将输入从 d_model 投影到 d_ff
        2. w3: 门控分支，将输入从 d_model 投影到 d_ff
        3. 对 w1 的输出应用 SiLU 激活函数
        4. 将激活后的结果与 w3 的输出逐元素相乘 (GLU 门控)
        5. w2: 将结果从 d_ff 投影回 d_model

    公式: output = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 SwiGLU 层

        Args:
            d_model: 模型隐藏层维度
            d_ff: 前馈网络内部维度，如果为 None 则自动计算为 (8/3)*d_model 向上取整到 64 的倍数
            device: 参数所在的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = swiglu_ffn(d_model, d_ff)

        # w1: 上投影层 (d_model -> d_ff)
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)

        # w2: 下投影层 (d_ff -> d_model)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, device=device, dtype=dtype)

        # w3: 门控投影层 (d_model -> d_ff)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播: 应用 SwiGLU

        公式: output = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T

        Args:
            x: 输入张量，形状为 (..., d_model)

        Returns:
            输出张量，形状为 (..., d_model)
        """
        # 上投影并应用 SiLU 激活
        # silu(x) = x * sigmoid(x)
        hidden = F.silu(self.w1(x))

        # 门控分支
        gate = self.w3(x)

        # 逐元素相乘 (GLU 门控机制)
        hidden = hidden * gate

        # 下投影
        output = self.w2(hidden)

        return output
