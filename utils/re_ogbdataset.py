from ogb.graphproppred import PygGraphPropPredDataset
import os.path as osp
import torch
import pandas as pd
from ogb.io.read_graph_pyg import read_graph_pyg
import numpy as np
from utils.re_tudataset import compute_lap_pe, compute_Ap_Up


class ReOGBDataset(PygGraphPropPredDataset):
    def __init__(
        self, root: str, name: str, pe_dim: int = 20, P: int = 4, **kwargs
    ) -> None:
        self.pe_dim = pe_dim
        self.P = P
        super().__init__(root=root, name = name, **kwargs)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        if self.meta_info["additional node files"] == "None":
            additional_node_files = []
        else:
            additional_node_files = self.meta_info["additional node files"].split(",")

        if self.meta_info["additional edge files"] == "None":
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info["additional edge files"].split(",")

        data_list = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=add_inverse_edge,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files,
            binary=self.binary,
        )

        # 处理标签
        if self.task_type == "subtoken prediction":
            graph_label_notparsed = pd.read_csv(
                osp.join(self.raw_dir, "graph-label.csv.gz"),
                compression="gzip",
                header=None,
            ).values
            graph_label = [
                str(graph_label_notparsed[i][0]).split(" ")
                for i in range(len(graph_label_notparsed))
            ]
            for i, g in enumerate(data_list):
                g.y = graph_label[i]
        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, "graph-label.npz"))[
                    "graph_label"
                ]
            else:
                graph_label = pd.read_csv(
                    osp.join(self.raw_dir, "graph-label.csv.gz"),
                    compression="gzip",
                    header=None,
                ).values

            has_nan = np.isnan(graph_label).any()
            for i, g in enumerate(data_list):
                if "classification" in self.task_type:
                    if has_nan:
                        g.y = (
                            torch.from_numpy(graph_label[i])
                            .view(1, -1)
                            .to(torch.float32)
                        )
                    else:
                        g.y = (
                            torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                        )
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        # 预处理
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # ====== 新增 LapPE 和 Ap/Up 计算 ======
        for data in data_list:
            data.pe = compute_lap_pe(
                data.edge_index, data.num_nodes, k=self.pe_dim, normalization="sym"
            )

            A_p_list, U_p_list = compute_Ap_Up(
                data.edge_index, data.num_nodes, P=self.P
            )

            for p in range(self.P):
                A_p = A_p_list[p]
                U_p = U_p_list[p]

                data[f"u_{p}_edge_index"] = U_p["idx"]
                data[f"u_{p}_edge_weight"] = U_p["wei"]

                data[f"a_{p}_edge_index"] = A_p["idx"]
                data[f"a_{p}_edge_weight"] = A_p["wei"]

        # ====== collate 并保存 ======
        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
