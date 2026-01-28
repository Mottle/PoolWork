import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU, LeakyReLU
from torch_geometric.nn import GENConv, DeepGCNLayer, global_mean_pool


class DeeperGCN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=14, dropout=0.5
    ):
        super(DeeperGCN, self).__init__()

        self.dropout = dropout
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()

        for i in range(num_layers):
            # 定义卷积算子 GENConv
            # aggr='softmax': 使用 SoftMax 聚合
            # learn_t=True: 学习温度参数 beta
            # learn_p=True: 如果使用 powermean，可以开启此项学习 p
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
                msg_norm=True,
                learn_msg_scale=True,
            )

            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = LeakyReLU(inplace=True)

            # block='res+': 使用 ResGCN+ 的残差连接方式 (DeeperGCN 推荐)
            layer = DeepGCNLayer(
                conv, norm, act, block="res+", dropout=dropout, ckpt_grad=False
            )

            self.layers.append(layer)
        self.final_norm = LayerNorm(hidden_channels)
        self.final_act = LeakyReLU(inplace=True)
        self.fin_map = Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index, batch=None, *args, **kwargs):
        x = self.node_encoder(x)

        # 逐层通过 DeeperGCN Layer
        # DeepGCNLayer 会自动处理 x = conv(norm(x)) + x 这种 Res+ 的逻辑
        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.final_norm(x)
        x = self.final_act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        global_feature = global_mean_pool(x, batch)
        # global_feature = self.fin_map(global_feature)

        return global_feature, 0
