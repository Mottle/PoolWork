import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class QuadraticLayer(nn.Module):
    """
    实现特殊的层：Y = X^T A X + WX + b

    其中:
    - X: 输入特征向量 (Batch_size, in_features)
    - A: 二次项权重矩阵 (in_features, in_features)
    - W: 线性项权重矩阵 (out_features, in_features)
    - b: 偏置向量 (out_features)
    - Y: 输出向量 (Batch_size, out_features)
    """
    def __init__(self, in_features: int, out_features: int):
        super(QuadraticLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. 二次项权重 A (A in X^T A X)
        # 为了处理 Batch_size，我们需要 out_features 个不同的 A 矩阵，
        # 每一个 A 矩阵负责计算 Y 的一个输出维度。
        # A 的形状: (out_features, in_features, in_features)
        self.A = nn.Parameter(torch.Tensor(out_features, in_features, in_features))

        # 2. 线性项权重 W (W in WX)
        # W 的形状: (out_features, in_features)
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))

        # 3. 偏置项 b
        # b 的形状: (out_features)
        self.b = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化参数
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

        # 初始化 A 矩阵
        # 可以使用 Xavier 或 Kaiming 初始化，但需要根据其三维结构调整。
        # 这里对 A 的每个 (in_features, in_features) 切片进行初始化
        for i in range(self.out_features):
            nn.init.kaiming_uniform_(self.A[i], a=math.sqrt(5))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Y = X^T A X + WX + b

        Args:
            X (torch.Tensor): 输入张量，形状通常为 (B, D_in)

        Returns:
            torch.Tensor: 输出张量，形状为 (B, D_out)
        """
        if X.dim() != 2:
            raise ValueError(f"输入 X 必须是二维张量 (Batch_size, in_features)，但接收到维度为 {X.dim()}")

        # --- 1. 计算线性项：WX + b ---
        # 线性运算 torch.nn.functional.linear(X, W, b)
        # W: (D_out, D_in), X: (B, D_in) -> 结果 (B, D_out)
        linear_output = F.linear(X, self.W, self.b)
        # linear_output: (B, D_out)


        # --- 2. 计算二次项：X^T A X ---
        # Y_j = X^T A_j X, 其中 A_j 是 A 的第 j 个切片 (D_in, D_in)

        A = (self.A + self.A.mT) / 2  # 确保 A 对称
        
        quadratic_output = torch.einsum('bi, oij, bj -> bo', X, A, X)
        # quadratic_output: (B, D_out)

        # --- 3. 最终输出：Y = X^T A X + WX + b ---
        final_output = quadratic_output + linear_output
        
        return final_output