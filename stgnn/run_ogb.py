import context
import torch
import numpy as np
from datetime import datetime
from constant import DATASET_PATH
from torch import nn
from torch_geometric.loader import DataLoader
from rich.progress import track
from perf_counter import get_time_sync
from benchmark_config import BenchmarkConfig
from utils.benchmark_result import BenchmarkResult
from torch.optim import Adam
import torch.nn.functional as F

# ==== 模型 ====
from classifier import Classifier
from span_tree_gnn import SpanTreeGNN
from span_tree_gnn_with_loss import SpanTreeGNNWithOrt
from dual_road_gnn import (
    DualRoadGNN,
    KFNDualRoadGNN,
    KRNDualRoadGNN,
    KFNDualRoadSTSplitGNN,
)
from dual_road_rev_attn_gnn import DualRoadRevAttnGNN
from baseline import BaseLine
from baseline_recent import BaseLineRc
from phop_baseline import HybirdPhopGNN
from st_split_gnn import SpanTreeSplitGNN
from kan_based_gin import KANBasedGIN
from graph_gps import GraphGPS
from sign import StackedSIGN
from h2gcn import H2GCN

# ==== OGB ====
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


# ============================================================
#  LOSS  （保留原逻辑，loss2 当前为 0 时等价于 loss1）
# ============================================================
def compute_loss(loss1, loss2):
    return loss1 + loss2 / (loss1 + loss2 + 1e-6).detach()


# ============================================================
#  TRAIN（自动适配单标签 / 多标签）
# ============================================================
def train_model(
    pooler,
    classifier,
    train_loader,
    optimizer,
    criterion,
    device,
    use_amp=False,
    dataset_name="ogbg-molhiv",
):
    pooler.train()
    classifier.train()

    total_loss = 0
    scaler = torch.amp.GradScaler(enabled=use_amp)
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    for data in track(train_loader, description=f"Run train", disable=True):
        optimizer.zero_grad()

        if data.x is None or data.x.size(1) <= 0:
            data.x = torch.ones((data.num_nodes, 1))
        data = data.to(device)

        if pooler.push_pe is not None:
            pe = data.pe.to(device)
            pooler.push_pe(pe)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            pooled, additional_loss = pooler(data.x, data.edge_index, data.batch, data)
            out = classifier(pooled)

            if dataset_name == "ogbg-molpcba":
                # 多标签任务，y ∈ {0,1,-1}^{N×T}，-1 表示未知，需要 mask
                y = data.y.to(torch.float32)
                is_labeled = y != -1

                # 将 -1 当成 0 参与 BCE，但用 mask 去掉其 loss 贡献
                target = (y > 0).float()

                per_entry_loss = F.binary_cross_entropy_with_logits(
                    out, target, reduction="none"
                )
                loss_main = (per_entry_loss * is_labeled).sum() / is_labeled.sum()
            else:
                # 单标签分类（molhiv / ppa）
                # OGB 的 y 通常是 [N,1]，CrossEntropy 需要 [N]
                y = data.y.view(-1)
                loss_main = criterion(out, y)

            loss = compute_loss(loss_main, additional_loss)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ============================================================
#  TEST（自动适配 ROC-AUC / AP / Accuracy）
# ============================================================
def test_model(
    pooler,
    classifier,
    test_loader,
    evaluator,
    device,
    use_amp=False,
    dataset_name="ogbg-molhiv",
):
    pooler.eval()
    classifier.eval()

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            if data.x is None or data.x.size(1) <= 0:
                data.x = torch.ones((data.num_nodes, 1))
            data = data.to(device)

            if pooler.push_pe is not None:
                pe = data.pe.to(device)
                pooler.push_pe(pe)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pooled, _ = pooler(data.x, data.edge_index, data.batch, data)
                out = classifier(pooled)

            y_true.append(data.y.cpu())

            if dataset_name == "ogbg-molhiv":
                # 取正类概率，shape: [N,1]
                prob = out.softmax(dim=1)[:, 1].unsqueeze(-1)
                y_pred.append(prob.cpu())

            elif dataset_name == "ogbg-molpcba":
                # 直接用 logits，evaluator 内部会处理
                y_pred.append(out.cpu())

            elif dataset_name == "ogbg-ppa":
                # multi-class，取 argmax 作为预测标签
                pred = out.argmax(dim=1).view(-1, 1)
                y_pred.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    if dataset_name == "ogbg-molhiv":
        return result["rocauc"]
    elif dataset_name == "ogbg-molpcba":
        return result["ap"]
    elif dataset_name == "ogbg-ppa":
        return result["acc"]
    else:
        raise ValueError(f"Unsupported dataset for test_model: {dataset_name}")


# ============================================================
#  RUN ONE SPLIT (OGB)
# ============================================================
def run_fold(
    dataset, loader, current_fold: int, config: BenchmarkConfig, dataset_name: str
):
    log_prefix = f"fold: {current_fold + 1}"
    print(f"{log_prefix} {dataset_name} 开始训练...")
    train_loader, val_loader, test_loader = loader

    stamp_start = get_time_sync()

    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    model, classifier = build_models(num_node_features, num_classes, config)

    # 选择合适的 loss（训练用）
    if dataset_name in ["ogbg-molhiv", "ogbg-ppa"]:
        criterion = nn.CrossEntropyLoss()
    elif dataset_name == "ogbg-molpcba":
        # 实际上我们在 train_model 内用 F.binary_cross_entropy_with_logits + mask
        # 这里传个占位符，保持接口统一
        criterion = None
    else:
        raise ValueError(f"Unknown dataset_name in run_fold: {dataset_name}")

    optimizer = build_optimizer(model, classifier, config)
    evaluator = Evaluator(name=dataset_name)

    epochs = config.epochs
    use_amp = config.amp

    val_res = BenchmarkResult()
    test_res = BenchmarkResult()

    no_increased_times = 0
    early_stop_epoch_num = config.early_stop_epochs

    for epoch in track(range(epochs), description=f"{log_prefix} 训练 {dataset_name}"):
        train_model(
            model,
            classifier,
            train_loader,
            optimizer,
            criterion,
            run_device,
            use_amp,
            dataset_name=dataset_name,
        )

        val_metric = test_model(
            model,
            classifier,
            val_loader,
            evaluator,
            run_device,
            use_amp,
            dataset_name=dataset_name,
        )
        test_metric = test_model(
            model,
            classifier,
            test_loader,
            evaluator,
            run_device,
            use_amp,
            dataset_name=dataset_name,
        )

        val_res.append(val_metric)
        test_res.append(test_metric)

        if config.early_stop:
            if val_metric < val_res.get_max():
                no_increased_times += 1
            else:
                no_increased_times = 0
            if no_increased_times >= early_stop_epoch_num:
                print(f"Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            metric_name = {
                "ogbg-molhiv": "ROC-AUC",
                "ogbg-molpcba": "AP",
                "ogbg-ppa": "Acc",
            }[dataset_name]
            print(
                f"{log_prefix}, Epoch {epoch + 1:03d} "
                f"Val {metric_name}: {val_metric:.4f}, "
                f"Test {metric_name}: {test_metric:.4f}"
            )

    stamp_end = get_time_sync()

    print(
        f"{log_prefix} {test_res.format()}\n"
        f"{log_prefix} {test_res.format(last=1)}\n"
        f"{log_prefix} {test_res.format(last=10)}\n"
        f"{log_prefix} {test_res.format(last=50)}\n"
        f"{log_prefix} 总运行时间: {(stamp_end - stamp_start) / 60:.4f}min"
    )

    return test_res


# ============================================================
#  BUILD MODELS（保持你的原有结构）
# ============================================================
def build_models(num_node_features, num_classes, config: BenchmarkConfig):
    input_dim = num_node_features
    hidden_dim = config.hidden_channels
    num_layers = config.num_layers
    model_type = config.model
    dropout = config.dropout

    if model_type == "mst":
        model = SpanTreeGNN(
            input_dim, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        ).to(run_device)
    elif model_type == "gcn" or model_type == "gin":
        model = BaseLine(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone=model_type,
            num_layers=num_layers,
            dropout=dropout,
        ).to(run_device)
    elif model_type == "mstort":
        model = SpanTreeGNNWithOrt(
            input_dim, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        ).to(run_device)
    elif model_type == "dualroad":
        model = DualRoadGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k=3
        ).to(run_device)
    elif model_type == "dualroad_kf":
        model = KFNDualRoadGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k=3
        ).to(run_device)
    elif model_type == "dualroad_kr":
        model = KRNDualRoadGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k=3
        ).to(run_device)
    elif model_type == "dualroad_rev_attn":
        model = DualRoadRevAttnGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        ).to(run_device)
    elif model_type == "dualroad_kf_sts":
        model = KFNDualRoadSTSplitGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k=3
        ).to(run_device)
    elif model_type == "st_split":
        model = SpanTreeSplitGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, num_splits=4
        ).to(run_device)
    elif model_type == "kan_gin":
        model = KANBasedGIN(input_dim, hidden_dim, num_layers=num_layers).to(run_device)
    elif model_type == "quad_gin":
        model = BaseLine(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="quad",
            num_layers=num_layers,
            dropout=dropout,
        ).to(run_device)
    elif model_type == "gt":
        model = BaseLine(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="gt",
            num_layers=num_layers,
            dropout=dropout,
        ).to(run_device)
    elif model_type == "phop_gcn":
        model = BaseLine(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="phop_gcn",
            num_layers=num_layers,
            dropout=dropout,
            embed=True,
        ).to(run_device)
    elif model_type == "phop_gin":
        model = BaseLine(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="phop_gin",
            num_layers=num_layers,
            dropout=dropout,
            embed=True,
        ).to(run_device)
    elif model_type == "phop_linkgcn":
        model = BaseLine(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="phop_linkgcn",
            num_layers=num_layers,
            dropout=dropout,
            embed=True,
        ).to(run_device)
    elif model_type == "hybird":
        model = HybirdPhopGNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, p=2, k=3
        ).to(run_device)
    elif model_type == "hybird_rw":
        model = HybirdPhopGNN(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            p=3,
            k=3,
            backbone="rw",
        ).to(run_device)
    elif model_type == "mix_hop":
        model = BaseLineRc(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="mix_hop",
            num_layers=num_layers,
            dropout=dropout,
            embed=True,
        ).to(run_device)
    elif model_type == "appnp":
        model = BaseLineRc(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="appnp",
            num_layers=num_layers,
            dropout=dropout,
            embed=True,
        ).to(run_device)
    elif model_type == "graph_gps":
        model = GraphGPS(
            input_dim, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        ).to(run_device)
    elif model_type == "sign":
        model = StackedSIGN(
            input_dim,
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            num_hops=3,
            dropout=dropout,
        ).to(run_device)
    elif model_type == "h2gcn":
        model = H2GCN(input_dim, hidden_dim, k=2, dropout=dropout).to(run_device)
    elif model_type == "gcn2":
        model = BaseLineRc(
            input_dim,
            hidden_dim,
            hidden_dim,
            backbone="gcn2",
            num_layers=num_layers,
            dropout=dropout,
            embed=True,
        ).to(run_device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    classifier = Classifier(hidden_dim, hidden_dim, num_classes).to(run_device)
    return model, classifier


# ============================================================
#  OPTIMIZER
# ============================================================
def build_optimizer(pooler, classifier, config: BenchmarkConfig):
    return Adam(list(pooler.parameters()) + list(classifier.parameters()), lr=config.lr)


# ============================================================
#  PROCESS DATASET (OGB 官方 split)
# ============================================================
def process_dataset(dataset, train_ids, val_ids, test_ids, batch_size):
    train_loader = DataLoader(dataset[train_ids], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_ids], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_ids], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ============================================================
#  RUN SINGLE SPLIT (OGB)
# ============================================================
def run_single_split(dataset, config: BenchmarkConfig, dataset_name: str):
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    train_loader, val_loader, test_loader = process_dataset(
        dataset, train_idx, val_idx, test_idx, config.batch_size
    )

    test_result = run_fold(
        dataset,
        (train_loader, val_loader, test_loader),
        current_fold=0,
        config=config,
        dataset_name=dataset_name,
    )

    all, last, last10, last50 = process_results([test_result])
    return all, last, last10, last50


# ============================================================
#  RESULT PROCESSING
# ============================================================
def format_result(mean, std):
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def process_results(results: list[BenchmarkResult]):
    last_res = [result.get_mean(1) for result in results]
    last_10_res = [result.get_mean(10) for result in results]
    last_50_res = [result.get_mean(50) for result in results]
    last_all_res = [result.get_mean() for result in results]

    return (
        (np.mean(last_all_res), np.std(last_all_res)),
        (np.mean(last_res), np.std(last_res)),
        (np.mean(last_10_res), np.std(last_10_res)),
        (np.mean(last_50_res), np.std(last_50_res)),
    )


# ============================================================
#  DATASETS (所有 graph-level OGB，除 code2)
# ============================================================
def datasets():
    # 可以自由增减，只要是 graph-level 且非 code2 即可
    names = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa"]
    for name in names:
        yield name, PygGraphPropPredDataset(name=name)


# ============================================================
#  SAVE RESULT
# ============================================================
def save_result(
    results, filename, spent_time, config: BenchmarkConfig, config_disp=False
):
    with open(filename, "a", encoding="utf-8") as f:
        if config_disp:
            f.write(f"{config.format()}")
        for name, all, last, last10, last50 in results:
            f.write(f"{name} all   : {format_result(all[0], all[1])}\n")
            f.write(f"{name} last  : {format_result(last[0], last[1])}\n")
            f.write(f"{name} last10: {format_result(last10[0], last10[1])}\n")
            f.write(f"{name} last50: {format_result(last50[0], last50[1])}\n")
        f.write(f"总运行时间: {spent_time / 60:.2f} min\n")


# ============================================================
#  RUN（遍历所有数据集）
# ============================================================
def run(config: BenchmarkConfig):
    config.apply_random_seed()

    results = []
    all_start = get_time_sync()
    id = -1

    for dataset_name, dataset in track(datasets(), description="All OGB Datasets"):
        id += 1
        dataset_start = get_time_sync()

        all, last, last10, last50 = run_single_split(dataset, config, dataset_name)
        results.append((dataset_name, all, last, last10, last50))

        dataset_end = get_time_sync()
        save_result(
            [(dataset_name, all, last, last10, last50)],
            f"./stgnn/result/{config.model}.txt",
            dataset_end - dataset_start,
            config,
            id == 0,
        )

    all_end = get_time_sync()
    print(f"{config.format()}\n")
    print(f"总运行时间: {(all_end - all_start) / 60:.2f} min")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    global run_device
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BenchmarkConfig()
    config.hidden_channels = 128
    config.num_layers = 3
    config.graph_norm = True
    config.batch_size = 128
    config.epochs = 500
    config.dropout = 0.1
    config.catch_error = False
    config.early_stop = True
    config.early_stop_epochs = 50
    config.seed = 0
    config.amp = True  # 开启混合精度

    models = ["hybird_rw"]
    for model in models:
        config.model = model
        run(config)
