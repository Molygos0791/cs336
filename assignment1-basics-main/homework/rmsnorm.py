import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)
    
    相比于标准的 LayerNorm，RMSNorm 省略了均值中心化操作，
    仅依赖于均方根进行归一化，从而提高了计算效率。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 RMSNorm 层

        Args:
            d_model: 模型的隐藏层维度
            eps: 用于数值稳定性的微小值
            device: 参数所在的设备
            dtype: 参数的数据类型
        """
        super().__init__()
        self.eps = eps
        
        # 创建可学习的缩放参数 (gamma)，初始化为全 1
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播：对输入张量执行 RMSNorm

        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, d_model)

        Returns:
            归一化后的张量，形状和数据类型与输入相同
        """
        # 1. 保存原始数据类型
        original_dtype = x.dtype
        
        # 2. 为了数值稳定性，将输入向上转型为 float32
        x_float = x.to(torch.float32)
        
        # 3. 计算均方值 (在最后一个维度 d_model 上计算，并保持维度以便广播)
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        
        # 4. 使用平方根倒数进行归一化 (rsqrt 计算比 1/sqrt 更快且更稳定)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        
        # 5. 将结果向下转型回原始的数据类型
        x_norm = x_norm.to(original_dtype)
        
        # 6. 应用可学习的缩放权重 self.weight
        return x_norm * self.weight
    

    