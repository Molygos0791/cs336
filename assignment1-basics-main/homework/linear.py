"""
Linear 模块实现

这个模块实现了一个简单的线性变换层（没有偏置）。
线性变换的公式是：output = x @ W.T

其中：
- x: 输入张量，形状为 (..., in_features)
- W: 权重矩阵，形状为 (out_features, in_features) (PyTorch 标准格式)
- output: 输出张量，形状为 (..., out_features)
"""

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum


class Linear(nn.Module):
    """
    线性变换模块（无偏置）

    这个类实现了 PyTorch 的 nn.Linear 类似的功能，但不包含偏置项。
    权重形状遵循 PyTorch 标准：(out_features, in_features)

    参数:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        device (torch.device | None): 参数存储的设备
        dtype (torch.dtype | None): 参数的数据类型
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化线性层

        Args:
            in_features: 输入的最后一个维度大小
            out_features: 输出的最后一个维度大小
            device: 参数所在的设备 (如 'cpu', 'cuda')
            dtype: 参数的数据类型 (如 torch.float32)
        """
        # 必须调用父类的构造函数
        super().__init__()

        # 保存输入输出维度
        self.in_features = in_features
        self.out_features = out_features

        # 创建权重参数
        # 使用 PyTorch 标准格式：(out_features, in_features)
        # 这样可以与 nn.Linear 的 state_dict 兼容
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # 使用截断正态分布初始化权重
        # 这是 PyTorch 线性层的标准初始化方式
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播：应用线性变换

        线性变换的数学公式是：
            output = x @ weight.T

        其中：
        - x 的形状是 (..., in_features)
        - weight 的形状是 (out_features, in_features)
        - output 的形状是 (..., out_features)

        Args:
            x: 输入张量，形状为 (..., in_features)

        Returns:
            输出张量，形状为 (..., out_features)
        """
        # 使用 einsum 实现：x @ weight.T
        # weight: (out_features, in_features), x: (..., in_features)
        # output: (..., out_features)
        return einsum(x, self.weight, "... inp, out inp -> ... out")
