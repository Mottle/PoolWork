from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.io import fs
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import os.path as osp
import torch
from torch_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops


class ReTUDataset(TUDataset):
    def __init__(self, root: str, name: str, **kwargs) -> None:
        super().__init__(root, name, **kwargs)

    #overwrite process
    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

def read_tu_data(
    folder: str,
    prefix: str,
) -> Tuple[Data, Dict[str, Tensor], Dict[str, int]]:
    files = fs.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [osp.basename(f)[len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attribute = torch.empty((batch.size(0), 0))
    if 'node_attributes' in names:
        node_attribute = read_file(folder, prefix, 'node_attributes')
        if node_attribute.dim() == 1:
            node_attribute = node_attribute.unsqueeze(-1)

    node_label = torch.empty((batch.size(0), 0))
    if 'node_labels' in names:
        node_label = read_file(folder, prefix, 'node_labels', torch.long)
        if node_label.dim() == 1:
            node_label = node_label.unsqueeze(-1)
        node_label = node_label - node_label.min(dim=0)[0]
        node_labels = list(node_label.unbind(dim=-1))
        node_labels = [one_hot(x) for x in node_labels]
        if len(node_labels) == 1:
            node_label = node_labels[0]
        else:
            node_label = torch.cat(node_labels, dim=-1)

    edge_attribute = torch.empty((edge_index.size(1), 0))
    if 'edge_attributes' in names:
        edge_attribute = read_file(folder, prefix, 'edge_attributes')
        if edge_attribute.dim() == 1:
            edge_attribute = edge_attribute.unsqueeze(-1)

    edge_label = torch.empty((edge_index.size(1), 0))
    if 'edge_labels' in names:
        edge_label = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_label.dim() == 1:
            edge_label = edge_label.unsqueeze(-1)
        edge_label = edge_label - edge_label.min(dim=0)[0]
        edge_labels = list(edge_label.unbind(dim=-1))
        edge_labels = [one_hot(e) for e in edge_labels]
        if len(edge_labels) == 1:
            edge_label = edge_labels[0]
        else:
            edge_label = torch.cat(edge_labels, dim=-1)

    x = cat([node_attribute, node_label])
    edge_attr = cat([edge_attribute, edge_label])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = int(edge_index.max()) + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attribute.size(-1),
        'num_node_labels': node_label.size(-1),
        'num_edge_attributes': edge_attribute.size(-1),
        'num_edge_labels': edge_label.size(-1),
    }

    return data, slices, sizes


def cat(seq: List[Optional[Tensor]]) -> Optional[Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return torch.cat(values, dim=-1) if len(values) > 0 else None

def split(data: Data, batch: Tensor) -> Tuple[Data, Dict[str, Tensor]]:
    node_slice = cumsum(torch.bincount(batch))

    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(torch.bincount(batch[row]))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, Tensor)
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, int(batch[-1]) + 2, dtype=torch.long)

    return data, slices

def read_file(
    folder: str,
    prefix: str,
    name: str,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

def read_txt_array(
    path: str,
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    import fsspec
    with fsspec.open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)

def parse_txt_array(
    src: List[str],
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    # 1. 初始化常量和类型转换函数
    empty = torch.empty(0, dtype=dtype)
    to_number = float if empty.is_floating_point() else int
    
    # 定义缺失值
    MISSING_VALUE = -1

    # 2. 预处理所有行，并找到最大长度 (max_length)
    processed_lines = []
    max_length = 0
    
    for line in src:
        # 移除行首尾空格，并按分隔符切分
        parts = line.strip().split(sep)[start:end]
        
        # 过滤空字符串（可能是由于连续分隔符或行尾分隔符造成）
        parsed_numbers = [to_number(x) for x in parts if x] 
        
        processed_lines.append(parsed_numbers)
        
        if len(parsed_numbers) > max_length:
            max_length = len(parsed_numbers)

    # 如果没有找到任何数据，返回空张量
    if max_length == 0:
        return empty.squeeze()

    # 3. 填充并转换为 Tensor
    # 使用列表推导式，对每个处理过的行进行填充
    tensor_data = []
    for line_data in processed_lines:
        current_length = len(line_data)
        
        # 计算需要填充的数量
        padding_needed = max_length - current_length
        
        # 填充到最大长度
        padded_line = line_data + [MISSING_VALUE] * padding_needed
        
        tensor_data.append(padded_line)

    # 4. 创建最终的 PyTorch Tensor
    return torch.tensor(tensor_data, dtype=dtype, device=device).squeeze()