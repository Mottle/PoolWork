import context
import torch
import numpy as np
from datetime import datetime
from constant import DATASET_PATH
from torch import nn
# from torch_geometric.datasets import TUDataset
from utils.re_tudataset import ReTUDataset
from torch_geometric.loader import DataLoader
from rich.progress import track
from perf_counter import get_time_sync, measure_time
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from benchmark_config import BenchmarkConfig
from utils.benchmark_result import BenchmarkResult
from torch.optim import Adam
from classifier import Classifier
from span_tree_gnn import SpanTreeGNN
from span_tree_gnn_with_loss import SpanTreeGNNWithOrt
from dual_road_gnn import DualRoadGNN, KFNDualRoadGNN, KRNDualRoadGNN, KFNDualRoadSTSplitGNN
from dual_road_rev_attn_gnn import DualRoadRevAttnGNN
from baseline import BaseLine
from baseline_recent import BaseLineRc
from phop_baseline import HybirdPhopGNN
from st_split_gnn import SpanTreeSplitGNN
from kan_based_gin import KANBasedGIN
from graph_gps import GraphGPS
from sign import StackedSIGN
from h2gcn import H2GCN

def compute_loss(loss1, loss2):
    return loss1 + loss2 / (loss1 + loss2 + 1e-6).detach()

def train_model(pooler, classifier, train_loader, optimizer, criterion, device):
    pooler.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0
    
    # all = len(train_loader)
    cnt = 0
    for data in track(train_loader, description=f"Run train", disable=True):
        # if not data.is_undirected():
        #     print('WARNING: dataset is NOT undirected!!!')
        cnt += 1
        optimizer.zero_grad()
        if data.x == None or data.x.size(1) <= 0:
            data.x = torch.ones((data.num_nodes, 1))
        data = data.to(device)

        #pe
        if pooler.push_pe is not None:
            pe = data.pe.to(device)
            pooler.push_pe(pe)

        pooled, additional_loss = pooler(data.x, data.edge_index, data.batch, data)
        out = classifier(pooled)
        loss = compute_loss(criterion(out, data.y), additional_loss)
        loss.backward()
        optimizer.step()

        # alrs.step(loss)
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    
    return total_loss / len(train_loader), correct / total

def test_model(pooler, classifier, test_loader, criterion, device):
    pooler.eval()
    classifier.eval()

    total_loss = 0
    correct = 0
    total = 0
    
    all = len(test_loader)
    with torch.no_grad():
        for data in track(test_loader,  description=f"Run test batch: {all}", disable=True):
            if data.x == None or data.x.size(1) <= 0:
                data.x = torch.ones((data.num_nodes, 1))
            data = data.to(device)

            if pooler.push_pe is not None:
                pe = data.pe.to(device)
                pooler.push_pe(pe)

            pooled, additional_loss = pooler(data.x, data.edge_index, data.batch, data)
            out = classifier(pooled)
            loss = compute_loss(criterion(out, data.y), additional_loss)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    return total_loss / len(test_loader), correct / total

def run_fold(dataset, loader, current_fold: int, config: BenchmarkConfig):
    log_prefix = f'fold: {current_fold+1}'
    print(f'{log_prefix} {dataset} 开始训练...')
    train_loader, val_loader, test_loader = loader

    stamp_start = get_time_sync()
    
    # 模型参数
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
    # 创建模型
    model, classifier = build_models(num_node_features, num_classes, config)
    
    # print(f"模型参数数量: {sum(p.numel() for p in pooler.parameters()) + sum(p.numel() for p in classifier.parameters())}")
    
    # 优化器和损失函数
    optimizer = build_optimizer(model, classifier, config)
    criterion = nn.CrossEntropyLoss()
    
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

    # global alrs
    # alrs = ALRS(optimizer)

    for epoch in track(range(epochs), description=f'{log_prefix} 训练 {dataset}'):
        train_loss, train_acc = train_model(model, classifier, train_loader, optimizer, criterion, run_device)
        val_loss, val_acc     = test_model(model, classifier, val_loader, criterion, run_device)
        test_loss, test_acc   = test_model(model, classifier, test_loader, criterion, run_device)

        if epoch > no_record_epoch_num:
            train_loss_list.append(train_loss)
            train_acc_res.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_res.append(val_acc)
            test_loss_list.append(test_loss)
            test_acc_res.append(test_acc)
        
        if config.early_stop and epoch > no_record_epoch_num:
            if val_acc < val_acc_res.get_max():
                no_increased_times += 1
            else:
                no_increased_times = 0
            if no_increased_times >= early_stop_epoch_num:
                print(f'Early stop at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'{log_prefix}, Epoch {epoch+1:03d} '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
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
    
    # 创建模型####
    # pooler = Pooler(input_dim, hidden_dim, num_layers=num_layers, pool_type=pooler_type, gnn_type=gnn_type, layer_norm=layer_norm).to(run_device)
    
    if model_type == 'mst':
        model = SpanTreeGNN(input_dim, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'gcn' or model_type == 'gin':
        model = BaseLine(input_dim, hidden_dim, hidden_dim, backbone=model_type, num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'mstort':
        model = SpanTreeGNNWithOrt(input_dim, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'dualroad':
        model = DualRoadGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k = 3).to(run_device)
    elif model_type == 'dualroad_kf':
        model = KFNDualRoadGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k = 3).to(run_device)
    elif model_type == 'dualroad_kr':
        model = KRNDualRoadGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k = 3).to(run_device)
    elif model_type == 'dualroad_rev_attn':
        model = DualRoadRevAttnGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'dualroad_kf_sts':
        model = KFNDualRoadSTSplitGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, k = 3).to(run_device)
    elif model_type == 'st_split':
        model = SpanTreeSplitGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, num_splits=4).to(run_device)
    elif model_type == 'kan_gin':
        model = KANBasedGIN(input_dim, hidden_dim, num_layers=num_layers).to(run_device)
    elif model_type == 'quad_gin':
        model = BaseLine(input_dim, hidden_dim, hidden_dim, backbone='quad', num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'gt':
        model = BaseLine(input_dim, hidden_dim, hidden_dim, backbone='gt', num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'phop_gcn':
        model = BaseLine(input_dim, hidden_dim, hidden_dim, backbone='phop_gcn', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)
    elif model_type == 'phop_gin':
        model = BaseLine(input_dim, hidden_dim, hidden_dim, backbone='phop_gin', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)
    elif model_type == 'phop_linkgcn':
        model = BaseLine(input_dim, hidden_dim, hidden_dim, backbone='phop_linkgcn', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)
    elif model_type == 'hybird':
        model = HybirdPhopGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, p = 2, k = 3).to(run_device)
    elif model_type == 'hybird_rw':
        model = HybirdPhopGNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, p = 3, k = 3, backbone='rw').to(run_device)
    elif model_type == 'mix_hop':
        model = BaseLineRc(input_dim, hidden_dim, hidden_dim, backbone='mix_hop', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)
    elif model_type == 'appnp':
        model = BaseLineRc(input_dim, hidden_dim, hidden_dim, backbone='appnp', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)
    # elif model_type == 'sign':
    #     model = BaseLineRc(input_dim, hidden_dim, hidden_dim, backbone='sign', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)
    # elif model_type == 'graph_gps':
    #     model = BaseLineRc(input_dim, hidden_dim, hidden_dim, backbone='graph_gps', num_layers=num_layers, dropout=dropout, embed=True, pos_emb=True).to(run_device)
    elif model_type == 'graph_gps':
        model = GraphGPS(input_dim, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout).to(run_device)
    elif model_type == 'sign':
        model = StackedSIGN(input_dim, hidden_dim, hidden_dim, num_layers=num_layers, num_hops=3, dropout=dropout).to(run_device)
    elif model_type == 'h2gcn':
        model = H2GCN(input_dim, hidden_dim, k = 2, dropout=dropout).to(run_device)
    elif model_type == 'gcn2':
        model = BaseLineRc(input_dim, hidden_dim, hidden_dim, backbone='gcn2', num_layers=num_layers, dropout=dropout, embed=True).to(run_device)

    classifier = Classifier(hidden_dim, hidden_dim, num_classes).to(run_device)

    return model, classifier

def build_optimizer(pooler, classifier, config: BenchmarkConfig):
    # return torch.optim.Adam(list(pooler.parameters()) + list(classifier.parameters()), lr=0.001)
    return Adam(list(pooler.parameters()) + list(classifier.parameters()), lr=config.lr)

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
    # mean_result, max_result = merge_results(results)
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

# def merge_results(results: list[BenchmarkResult]):
#     mean_result = BenchmarkResult()
#     max_result = BenchmarkResult()
#     for result in results:
#         mean_result.append(result.get_mean())
#         max_result.append(result.get_max())

#     return mean_result, max_result

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
    # for result in results:
    #     all_mean += result.get_mean() / size
    #     last_mean += result.get_mean(1) / size
    #     last_10_mean += result.get_mean(10) / size
    #     last_50_mean += result.get_mean(50) / size
    #     all_std += result.get_std() / size
    #     last_std += result.get_std(1) / size
    #     last_10_std += result.get_std(10) / size
    #     last_50_std += result.get_std(50) / size

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
            # 'MSRC_21'
        ]
    elif sets == 'common':
        datasets = [
            # 'DD',
            'PROTEINS',
            'NCI1',
            'NCI109',
            'COX2',
            'IMDB-BINARY',
            'IMDB-MULTI',
            'FRANKENSTEIN',
            'COLLAB',
            # 'REDDIT-BINARY',
            # 'Synthie',
            # 'SYNTHETIC',
            # 'MSRC_9',
            # 'MSRC_21',
        ]
    elif sets == 'dense':
        datasets = [
            'mit_ct1', #d≈146.92
            'mit_ct2',
            # 'COLLAB', #d≈66
            'highschool_ct1', #d≈20.8
            'highschool_ct2',
            'infectious_ct1', #d≈18.39
            'infectious_ct2',
        ]
    for i in range(len(datasets)):
        yield ReTUDataset(root=DATASET_PATH, name=datasets[i])

def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# def format_result(result):
#     try:
#         (ds, mean_result, max_result) = result
#         return f'数据集: {ds}, mean: {mean_result.get_mean()}, max: {max_result.get_max()}'
#     except:
#         return 'NONE'


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
        
        # if config.use_simple_datasets:
        #     save_result([(f'{dataset}', all, last, last10, last50)], f'./stgnn/result_simple/{config.model}.txt', dataset_end - dataset_start, config, id == 0)
        # else:
        save_result([(f'{dataset}', all, last, last10, last50)], f'./stgnn/result/{config.model}.txt', dataset_end - dataset_start, config, id == 0)
    all_end = get_time_sync()

    print(f'{config.format()}\n')

    # for result in results:
    #     print(format_result(result[0], result[1]))
    print(f'总运行时间: {(all_end - all_start) / 60:.2f} min')

if __name__ == '__main__':
    global run_device
    run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # now = datetime.now()
    # now_str = now.strftime('%Y-%m-%d-%H-%M')

    config = BenchmarkConfig()
    config.hidden_channels = 128
    config.num_layers = 3
    config.graph_norm = True
    config.batch_size = 128
    config.epochs = 500
    config.dropout = 0.1
    # config.use_simple_datasets = False
    config.sets = 'common'
    config.catch_error = False
    config.early_stop = True
    config.early_stop_epochs = 50
    config.seed = None
    config.kfold = 10

    models = ['gcn2']
    # models = ['topk']
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
    