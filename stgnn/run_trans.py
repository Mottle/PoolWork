import context
import torch
import numpy as np
from datetime import datetime
from constant import DATASET_PATH
from torch import nn
from utils import pre_trans
from utils.re_tudataset import ReTUDataset
from torch_geometric.loader import DataLoader
from rich.progress import track
from perf_counter import get_time_sync, measure_time
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from benchmark_config import BenchmarkConfig
from utils.benchmark_result import BenchmarkResult
from torch.optim import Adam

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from classifier import Classifier
from span_tree_gnn import SpanTreeGNN
from true_gps import GraphGPS
from san import SAN
from utils.pre_trans import pre_transform_all
from sat import SAT
from grit import GRIT

def compute_loss(loss1, loss2):
    return loss1 + loss2 / (loss1 + loss2 + 1e-6).detach()

# def train_model(pooler, classifier, train_loader, optimizer, criterion, device, use_amp=False):
#     pooler.train()
#     classifier.train()

#     total_loss = 0
#     correct = 0
#     total = 0
    
#     # 1. 初始化 Scaler (仅在开启 AMP 且是 CUDA 设备时生效)
#     scaler = torch.amp.GradScaler(enabled=use_amp)
    
#     # 获取设备类型字符串 (用于 autocast)
#     device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
#     for data in track(train_loader, description=f"Run train", disable=True):
#         optimizer.zero_grad()
#         if data.x is None or data.x.size(1) <= 0:
#             data.x = torch.ones((data.num_nodes, 1))
#         data = data.to(device)

#         # 2. 前向传播使用 autocast 上下文
#         with torch.amp.autocast(device_type=device_type, enabled=use_amp):
#             pe = data.pe if hasattr(data, 'pe') else None
#             pooled, additional_loss = pooler(data.x, data.edge_index, data.batch, pe=pe)
#             out = classifier(pooled)
#             loss = compute_loss(criterion(out, data.y), additional_loss)

#         # 3. 反向传播与优化器更新
#         if use_amp:
#             # 缩放损失并反向传播
#             scaler.scale(loss).backward()
#             # scaler.step() 会先取消缩放梯度，如果梯度无 inf/NaN，则调用 optimizer.step()
#             scaler.step(optimizer)
#             # 更新缩放因子
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()
        
#         # 统计逻辑保持不变
#         total_loss += loss.item()
#         pred = out.argmax(dim=1)
#         correct += (pred == data.y).sum().item()
#         total += data.y.size(0)
    
#     return total_loss / len(train_loader), correct / total

def train_model(pooler, classifier, train_loader, optimizer, criterion, device, use_amp=False):
    pooler.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.amp.GradScaler(enabled=use_amp)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    for data in track(train_loader, description=f"Run train", disable=True):
        optimizer.zero_grad()
        if data.x is None or data.x.size(1) <= 0:
            data.x = torch.ones((data.num_nodes, 1))
        data = data.to(device)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            pe = data.pe if hasattr(data, 'pe') else None
            # 注意：如果你的模型 forward 接收 pe，确保这里传了
            pooled = pooler(data.x, data.edge_index, data.batch, pe=pe, spd = data.spd_attr, rrwp = data.rrwp, rrwp_abs=data.rrwp_abs) 
            # 如果 pooler 返回的是 (out, loss)，请自行解包，这里假设 GraphGPS 只返回 logits
            # 根据上文 GraphGPS 代码，它只返回 out。如果你的接口有变化请调整。
            if isinstance(pooled, tuple):
                pooled, additional_loss = pooled
            else:
                additional_loss = 0
            
            out = classifier(pooled)
            loss = compute_loss(criterion(out, data.y), additional_loss)

        if use_amp:
            scaler.scale(loss).backward()
            # --- 新增: 梯度裁剪 (Gradient Clipping) ---
            scaler.unscale_(optimizer) # 裁剪前必须 unscale
            torch.nn.utils.clip_grad_norm_(pooler.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            # ----------------------------------------
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # --- 新增: 梯度裁剪 (Gradient Clipping) ---
            torch.nn.utils.clip_grad_norm_(pooler.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            # ----------------------------------------
            optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    
    return total_loss / len(train_loader), correct / total

def test_model(pooler, classifier, test_loader, criterion, device, use_amp=False):
    pooler.eval()
    classifier.eval()

    total_loss = 0
    correct = 0
    total = 0
    
    # 获取设备类型字符串
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    all_batches = len(test_loader)
    with torch.no_grad():
        for data in track(test_loader, description=f"Run test batch: {all_batches}", disable=True):
            if data.x is None or data.x.size(1) <= 0:
                data.x = torch.ones((data.num_nodes, 1))
            data = data.to(device)

            # 在推理阶段也开启 autocast 以保持精度/性能策略与训练一致
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pe = data.pe if hasattr(data, 'pe') else None
                pooled, additional_loss = pooler(data.x, data.edge_index, data.batch, pe=pe, spd =  data.spd_attr, rrwp = data.rrwp, rrwp_abs=data.rrwp_abs)
                out = classifier(pooled)
                loss = compute_loss(criterion(out, data.y), additional_loss)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    return total_loss / len(test_loader), correct / total

# --- 构建调度器 ---
def build_scheduler(optimizer, config: BenchmarkConfig):
    warmup_epochs = config.warmup_epochs if hasattr(config, 'warmup_epochs') else 10
    print(f"Using Warmup + Cosine Annealing Scheduler (Warmup: {warmup_epochs} epochs)")
    
    # 1. Warmup 阶段: 学习率从 lr * start_factor 线性增加到 lr
    # start_factor 设为 1e-4 意味着从极小的 lr 开始预热
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-4, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # 2. Cosine 阶段: 学习率从 lr 按余弦曲线下降到 min_lr (这里设为 lr * 1e-2)
    # T_max 为剩下的 epoch 数
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs - warmup_epochs, 
        eta_min=config.lr * 1e-2
    )
    
    # 3. 串联两个阶段
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    return scheduler, 'epoch' # 返回调度器和更新频率类型


def run_fold(dataset, loader, current_fold: int, config: BenchmarkConfig):
    log_prefix = f'fold: {current_fold+1}'
    print(f'{log_prefix} {dataset} 开始训练...')
    train_loader, val_loader, test_loader = loader

    stamp_start = get_time_sync()
    
    # 模型参数
    num_node_features = max(1, dataset.num_node_features)
    num_classes = dataset.num_classes
    
    # 创建模型
    model, classifier = build_models(num_node_features, num_classes, config)
    
    # 优化器
    optimizer = build_optimizer(model, classifier, config)
    criterion = nn.CrossEntropyLoss()
    
    # --- 修改: 获取调度器 ---
    scheduler, scheduler_type = build_scheduler(optimizer, config)
    
    # 训练循环
    epochs = config.epochs

    train_loss_list = []
    train_acc_res = BenchmarkResult()
    val_loss_list = []
    val_acc_res = BenchmarkResult()
    test_loss_list = []
    test_acc_res = BenchmarkResult()
    
    no_increased_times = 0
    no_record_epoch_num = 0
    early_stop_epoch_num = config.early_stop_epochs
    use_amp = config.amp

    for epoch in track(range(epochs), description=f'{log_prefix} 训练 {dataset}'):
        train_loss, train_acc = train_model(model, classifier, train_loader, optimizer, criterion, run_device, use_amp=use_amp)
        val_loss, val_acc     = test_model(model, classifier, val_loader, criterion, run_device, use_amp=use_amp)
        test_loss, test_acc   = test_model(model, classifier, test_loader, criterion, run_device, use_amp=use_amp)

        # --- 修改: 调度器步进逻辑 ---
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            # Warmup/Cosine 通常每个 epoch 调用一次，不需要 metric
            scheduler.step()

        if epoch > no_record_epoch_num:
            train_loss_list.append(train_loss)
            train_acc_res.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_res.append(val_acc)
            test_loss_list.append(test_loss)
            test_acc_res.append(test_acc)
        
        # --- 修改: Early Stop 逻辑 (Warmup 期间不早停) ---
        if config.early_stop and epoch > no_record_epoch_num:
            # 获取 warmup 轮数，如果是 plateau 则为 0
            min_warmup_epochs = config.warmup_epochs if (hasattr(config, 'warmup_epochs') and scheduler_type == 'epoch') else 0
            
            # 计算是否提升
            if val_acc < val_acc_res.get_max():
                no_increased_times += 1
            else:
                no_increased_times = 0
            
            # 只有当当前 epoch 超过 warmup 期，且未提升次数超过阈值时，才停止
            if epoch > min_warmup_epochs and no_increased_times >= early_stop_epoch_num:
                print(f'Early stop at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f'{log_prefix}, Epoch {epoch+1:03d} '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, lr: {current_lr:.6f}')
    
    stamp_end = get_time_sync()

    print(f'{log_prefix} {test_acc_res.format()}\n' +
          f'{log_prefix} {test_acc_res.format(last=1)}\n' +
          f'{log_prefix} {test_acc_res.format(last=10)}\n' +
          f'{log_prefix} {test_acc_res.format(last=50)}\n' +
          f'{log_prefix} 总运行时间: {(stamp_end - stamp_start) / 60:.4f}min')
    return train_acc_res, test_acc_res

def build_models(num_node_features, num_classes, config: BenchmarkConfig):
    input_dim = num_node_features
    hidden_dim = config.hidden_channels
    num_layers = config.num_layers
    model_type = config.model
    layer_norm = config.graph_norm
    dropout = config.dropout
        
    if model_type == 'graph_gps':
        model = GraphGPS(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, pe_dim=20).to(run_device)
    elif model_type == 'san':
        model = SAN(in_dim=input_dim,edge_dim=4, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'sat':
        model = SAT(in_size=input_dim, d_model=hidden_dim, pe_dim=20, abs_pe=True, global_pool='cls').to(run_device)
    elif model_type == 'grit':
        model = GRIT(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            # heads=4,
            dropout=dropout,
        ).to(run_device)
    else:
        raise ValueError(f"模型类型无效: {model_type}")

    classifier = Classifier(hidden_dim, hidden_dim, num_classes).to(run_device)

    return model, classifier

# --- 修改: 构建优化器 ---
def build_optimizer(pooler, classifier, config: BenchmarkConfig):
    params = list(pooler.parameters()) + list(classifier.parameters())
    
    # 尝试从配置中获取 weight_decay，默认 0
    weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0
    
    # Graph Transformer 推荐使用 AdamW
    if config.model in ['graph_gps', 'gt', 'transformer', 'sign']:
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=weight_decay)
    
    return Adam(params, lr=config.lr, weight_decay=weight_decay)

#对模型在dataset上运行k-fold
def run_k_fold4dataset(dataset, config: BenchmarkConfig):
    kfold_dataset = kfold_split(dataset, config.kfold, config.seed)
    results = []
    all_start = get_time_sync()

    for fold, (train_idx, test_idx) in track(enumerate(kfold_dataset), total=config.kfold, description='k-fold'):
        print(f'Run model: {config.model}')
        train_idx, val_idx = split_train_val(train_idx, config.kfold)
        train_loader, val_loader, test_loader = process_dataset(dataset, train_idx, val_idx, test_idx, config.batch_size)
        if config.catch_error:
            try:
                train_result, test_result = run_fold(dataset, (train_loader, val_loader, test_loader), current_fold=fold, config= config)
                results.append(test_result)
            except Exception as e:
                print(f'fold-{fold} 运行 {dataset} 时出错: {e}')
        else:
            train_result, test_result = run_fold(dataset, (train_loader, val_loader, test_loader), current_fold=fold, config= config)
            results.append(test_result)
    all_end = get_time_sync()

    #合并k-fold实验结果
    all, last, last_10, last_50 = process_results(results)
    print(
        '----------RESULTS----------\n',
        f'spend time: {(all_end - all_start) / 60:.2f} min\n',
        f'{dataset} for {config.kfold} fold:\n', 
        f'all     : {format_result(all[0], all[1])}\n',
        f'last-50 : {format_result(last_50[0], last_50[1])}\n',
        f'last-10 : {format_result(last_10[0], last_10[1])}\n',
        f'last    : {format_result(last[0], last[1])}\n',
        '---------------------------\n'
    )

    return all, last, last_10, last_50

def format_result(mean, std):
    return f'{mean * 100:.2f}% ± {std * 100:.2f}%'

def process_results(results: list[BenchmarkResult]):
    size = float(len(results))
    all_mean, all_std = 0, 0
    last_mean, last_std = 0, 0
    last_10_mean, last_10_std = 0, 0
    last_50_mean, last_50_std = 0, 0

    last_res = [result.get_mean(1) for result in results]
    last_10_res = [result.get_mean(10) for result in results]
    last_50_res = [result.get_mean(50) for result in results]
    last_all_res = [result.get_mean() for result in results]

    all_mean = np.mean(last_all_res)
    all_std = np.std(last_all_res)
    last_mean = np.mean(last_res)
    last_std = np.std(last_res)
    last_10_mean = np.mean(last_10_res)
    last_10_std = np.std(last_10_res)
    last_50_mean = np.mean(last_50_res)
    last_50_std = np.std(last_50_res)

    return (all_mean, all_std), (last_mean, last_std), (last_10_mean, last_10_std), (last_50_mean, last_50_std)

def split_train_val(train_idx, kfold):
    val_ratio = 1.0 / float(kfold - 1)
    permuted = np.random.permutation(train_idx)
    val_size = int(len(permuted) * val_ratio)
    val_idx = permuted[:val_size].tolist()
    new_train_idx = permuted[val_size:].tolist()
    return new_train_idx, val_idx

def process_dataset(dataset, train_ids, val_ids, test_ids, batch_size):
    train_ids = np.random.permutation(train_ids).tolist()
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    return train_loader, val_loader, test_loader

def kfold_split(source_dataset, k, seed):
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    return kfold.split(source_dataset)

def datasets(sets='common'):
    datasets = []
    if sets == 'simple':
        datasets = [
            'NCI1',
            'COX2',
            'IMDB-BINARY',
        ]
    elif sets == 'common':
        datasets = [
            'DD',
            'PROTEINS',
            'NCI1',
            'NCI109',
            'COX2',
            'IMDB-BINARY',
            'IMDB-MULTI',
            'FRANKENSTEIN',
        ]
    elif sets == 'com':
        datasets = [
            'IMDB-BINARY',
            'IMDB-MULTI',
            # 'REDDIT-BINARY',
            # 'COLLAB',
        ]
    elif sets == 'bio&chem':
        datasets = [
            # 'DD',
            # 'PROTEINS',
            # 'NCI1',
            'NCI109',
            'COX2',
            'FRANKENSTEIN'
        ]
    for i in range(len(datasets)):
        yield ReTUDataset(root=DATASET_PATH, name=datasets[i], pre_transform=pre_transform_all())

def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_result(results, filename, spent_time, config: BenchmarkConfig, config_disp = False):
    with open(filename, 'a', encoding='utf-8') as f:
        if len(results) == 0:
            print('save_result empty')
            return
        if config_disp:
            f.write(f'{config.format()}')
        for (name, all, last, last10, last50) in results:
            f.write(f'{name} all   : {format_result(all[0], all[1])}\n')
            f.write(f'{name} last  : {format_result(last[0], last[1])}\n')
            f.write(f'{name} last10: {format_result(last10[0], last10[1])}\n')
            f.write(f'{name} last50: {format_result(last50[0], last50[1])}\n')
        f.write(f'总运行时间: {spent_time / 60:.2f} min\n')
        f.close()

#对模型运行k-fold - dataset 
def run(config: BenchmarkConfig):
    config.apply_random_seed()

    results = []
    all_start = get_time_sync()
    id = -1
    for dataset in track(datasets(config.sets), description="All Datasets"):
        id += 1
        dataset_start = get_time_sync()
        if config.catch_error:
            try:
                # result = run_fold(dataset, config)
                all, last, last10, last50 = run_k_fold4dataset(dataset, config)
                results.append((f'{dataset}', all, last, last10, last50))
            except Exception as e:
                print(f'运行 {dataset} 时出错: {e}')
                results.append((f'{dataset}', BenchmarkResult(), BenchmarkResult()))
                continue
        else:
            all, last, last10, last50  = run_k_fold4dataset(dataset, config)
            results.append((f'{dataset}', all, last, last10, last50))
        dataset_end = get_time_sync()
        
        save_result([(f'{dataset}', all, last, last10, last50)], f'./stgnn/result_fin/{config.model}.txt', dataset_end - dataset_start, config, id == 0)
    all_end = get_time_sync()

    print(f'{config.format()}\n')
    print(f'总运行时间: {(all_end - all_start) / 60:.2f} min')

if __name__ == '__main__':
    global run_device
    run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = BenchmarkConfig()
    config.hidden_channels = 64
    config.num_layers = 3
    config.graph_norm = False
    config.batch_size = 64
    config.epochs = 200
    config.dropout = 0.1
    # config.sets = 'com'
    config.sets = 'bio&chem'
    config.catch_error = False
    config.early_stop = True
    config.early_stop_epochs = 50
    config.seed = None
    config.kfold = 10
    config.lr = 5e-4
    
    # --- 新增: Warmup 和 Weight Decay 配置 ---
    config.warmup_epochs = 20  # 设置预热 epoch 数
    config.weight_decay = 1e-4 # Transformer 建议设置权重衰减
    config.amp = True         # 可根据显存情况开启 AMP
    # ---------------------------------------

    # 将 GraphGPS 加入测试列表
    models = ['san', 'sat']
    # models = ['gcn', 'gin', 'gat', 'mix_hop', 'appnp', 'gcn2']
    
    seeds = [0, 114514, 1919810, 77777]
    for model in models:
        config.model = model
        config.seed = seeds[0]

        if config.catch_error:
            try:
                run(config)
            except Exception as e:
                print(f'运行 {config.model} 时出错: {e}')
        else:
            run(config)