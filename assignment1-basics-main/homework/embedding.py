"""
Embedding 模块实现

这个模块实现了一个词嵌入层，用于将离散的 token ID 映射到连续的向量空间。
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    """
    词嵌入模块

    这个类实现了 PyTorch 的 nn.Embedding 类似的功能。

    参数:
        num_embeddings (int): 词汇表大小
        embedding_dim (int): 嵌入向量的维度 (d_model)
        device (torch.device | None): 参数存储的设备
        dtype (torch.dtype | None): 参数的数据类型
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化嵌入层

        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入向量的维度
            device: 参数所在的设备 (如 'cpu', 'cuda')
            dtype: 参数的数据类型 (如 torch.float32)
        """
        # 必须调用父类的构造函数
        super().__init__()

        # 保存参数
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 创建嵌入矩阵参数
        # 形状为 (num_embeddings, embedding_dim)，即 (vocab_size, d_model)
        # d_model 是最后一个维度
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # 使用截断正态分布初始化权重
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        前向传播：查找 token ID 对应的嵌入向量

        Args:
            token_ids: token ID 张量，形状为 (...)，可以是任意形状

        Returns:
            嵌入向量张量，形状为 (..., embedding_dim)
            即在输入的最后增加一个 embedding_dim 维度
        """
        # 使用索引操作查找嵌入向量
        # self.weight 的形状是 (num_embeddings, embedding_dim)
        # token_ids 的形状是 (...)
        # self.weight[token_ids] 的形状是 (..., embedding_dim)
        return self.weight[token_ids]
