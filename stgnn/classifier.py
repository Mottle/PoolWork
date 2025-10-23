import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.liner0 = torch.nn.Linear(in_dim, hidden_dim)
        self.liner1 = torch.nn.Linear(hidden_dim, num_classes)

        # self._initialize_weights()
    
    # def _initialize_weights(self):
    #     # Kaiming/He 初始化（适合ReLU族激活函数）
    #     torch.nn.init.kaiming_uniform_(self.liner0.weight)
    #     torch.nn.init.kaiming_uniform_(self.liner1.weight)
    
    #     # 初始化偏置为0
    #     if self.liner0.bias is not None:
    #         nn.init.constant_(self.liner0.bias, 0)
    #     if self.liner1.bias is not None:
    #         nn.init.constant_(self.liner1.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.liner0(x))
        x = self.liner0(x)
        # x = activate(x)
        x = F.leaky_relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.liner1(x)

        return x