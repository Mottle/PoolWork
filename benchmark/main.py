import context
import torch
import numpy as np
from constant import DATASET_PATH
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import Pooler, Classifier
from rich.progress import track
from perf_counter import get_time_sync, measure_time
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from benchmark_config import BenchmarkConfig
from benchmark_result import BenchmarkResult

def train_model(pooler, classifier, train_loader, optimizer, criterion, device):
    pooler.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0
    
    # all = len(train_loader)
    cnt = 0
    for data in track(train_loader, description=f"Run train", disable=True):
        cnt += 1
        optimizer.zero_grad()
        if data.x == None or data.x.size(1) <= 0:
            data.x = torch.ones((data.num_nodes, 1))
        data = data.to(device)
        pooled, additional_loss = pooler(data.x, data.edge_index, data.batch)
        out = classifier(pooled)
        loss = criterion(out, data.y) + additional_loss
        loss.backward()
        optimizer.step()
        
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
            pooled, _ = pooler(data.x, data.edge_index, data.batch)
            out = classifier(pooled)
            loss = criterion(out, data.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    return total_loss / len(test_loader), correct / total

def run_fold(dataset, train_loader, test_loader, current_fold: int, config: BenchmarkConfig):
    log_prefix = f'fold: {current_fold+1}'
    print(f'{log_prefix} {dataset} 开始训练...')

    stamp_start = get_time_sync()
    
    # 数据集信息
    # print(f"数据集: {train_dataset}")
    
    # train_size = len(train_dataset)
    # test_size = len(test_dataset)
    
    # print(f"训练集大小: {train_size}")
    # print(f"测试集大小: {test_size}")
    
    # 数据加载器
    # batch_size = config['batch_size']
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型参数
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
    # 创建模型
    pooler, classifier = build_models(num_node_features, num_classes, config)
    
    # print(f"模型参数数量: {sum(p.numel() for p in pooler.parameters()) + sum(p.numel() for p in classifier.parameters())}")
    
    # 优化器和损失函数
    optimizer = build_optimizer(pooler, classifier, config)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    epochs = config.epochs

    train_loss_list = []
    train_acc_res = BenchmarkResult()
    test_loss_list = []
    test_acc_res = BenchmarkResult()
    
    no_increased_times = 0
    no_record_epoch_num = 0
    early_stop_epoch_num = 30

    for epoch in track(range(epochs), description=f'{log_prefix} 训练 {dataset}'):
        train_loss, train_acc = train_model(pooler, classifier, train_loader, optimizer, criterion, run_device)
        test_loss, test_acc = test_model(pooler, classifier, test_loader, criterion, run_device)

        if epoch > no_record_epoch_num:
            train_loss_list.append(train_loss)
            train_acc_res.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_res.append(test_acc)
        
        if config.early_stop and epoch > no_record_epoch_num:
            if test_acc < test_acc_res.get_max():
                no_increased_times += 1
            else:
                no_increased_times = 0
            if no_increased_times >= early_stop_epoch_num:
                print(f'Early stop at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'{log_prefix}, Epoch {epoch+1:03d}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    stamp_end = get_time_sync()

    print(f'{log_prefix} {test_acc_res.format()}')
    print(f'{log_prefix} 总运行时间: {(stamp_end - stamp_start) / 60:.4f}min\n')
    return train_acc_res, test_acc_res

def build_models(num_node_features, num_classes, config: BenchmarkConfig):
    input_dim = num_node_features
    hidden_dim = config.hidden_channels
    num_layers = config.num_layers
    pooler_type = config.pooler
    gnn_type = config.backbone
    layer_norm = config.graph_norm
    
    # 创建模型
    pooler = Pooler(input_dim, hidden_dim, num_layers=num_layers, pool_type=pooler_type, gnn_type=gnn_type, layer_norm=layer_norm).to(run_device)
    classifier = Classifier(pooler.get_out_dim(), pooler.get_out_dim(), num_classes).to(run_device)

    return pooler, classifier

def build_optimizer(pooler, classifier, config: BenchmarkConfig):
    return torch.optim.Adam(list(pooler.parameters()) + list(classifier.parameters()), lr=0.001)

#对模型在dataset上运行k-fold
def run_k_fold4dataset(dataset, config: BenchmarkConfig):
    kfold_dataset = kfold_split(dataset, config.kfold, config.seed)
    results = []
    all_start = get_time_sync()

    for fold, (train_idx, test_idx) in track(enumerate(kfold_dataset), total=config.kfold, description='k-fold'):
        print(f'Run model: {config.pooler}')
        train_loader, test_loader = process_dataset(dataset, train_idx, test_idx, config.batch_size)
        if config.catch_error:
            try:
                train_result, test_result = run_fold(dataset, train_loader, test_loader, current_fold=fold, config= config)
                results.append(test_result)
            except Exception as e:
                print(f'fold-{fold} 运行 {dataset} 时出错: {e}')
        else:
            train_result, test_result = run_fold(dataset, train_loader, test_loader, current_fold=fold, config= config)
            results.append(test_result)
    all_end = get_time_sync()

    #合并k-fold实验结果
    result = merge_results(results)
    print(f'总运行时间: {(all_end - all_start) / 60:.2f} min\n')
    print(f'{dataset}-{config.kfold}fold: {result.format()}')

    return result

def merge_results(results: list[BenchmarkResult]) -> BenchmarkResult:
    new = BenchmarkResult()
    for result in results:
        new.merge(result)
    return new


def process_dataset(dataset, train_ids, test_ids, batch_size):
    train_ids = np.random.permutation(train_ids).tolist()
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    return train_loader, test_loader

def kfold_split(source_dataset, k, seed):
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    return kfold.split(source_dataset)

def datasets(simple=False):
    if simple:
        datasets = [
            'NCI1',
            'COX2',
            'IMDB-BINARY',
        ]
        for i in range(len(datasets)):
            yield TUDataset(root=DATASET_PATH, name=datasets[i])
    else:
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
            'Synthie',
            'SYNTHETIC',
            # 'COIL-DEL',
            # 'Fingerprint'
            'MSRC_9',
            'MSRC_21',
        ]
        for i in range(len(datasets)):
            yield TUDataset(root=DATASET_PATH, name=datasets[i])

def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_result(result):
    return f'数据集: {result[0]}, 测试准确率: {result[1] * 100:.2f}% ± {result[2] * 100:.2f}%, 最佳测试准确率: {result[3] * 100:.2f}%'

def save_result(results, filename, spent_time, config: BenchmarkConfig):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f'{config.format()}')
        for (name, result) in results:
            f.write(f'{name}: {result.format()}\n')
        f.write(f'总运行时间: {spent_time / 60:.2f} min\n')

#对模型运行k-fold - dataset
def run(config: BenchmarkConfig):
    results = []
    all_start = get_time_sync()
    for dataset in track(datasets(config.use_simple_datasets), description="All Datasets"):
        if config.catch_error:
            try:
                # result = run_fold(dataset, config)
                result = run_k_fold4dataset(dataset, config)
                results.append(result)
            except Exception as e:
                print(f'运行 {dataset} 时出错: {e}')
                results.append((dataset, -1, -1))
                continue
        else:
            result = run_k_fold4dataset(dataset, config)
            results.append((f'{dataset}', result))
    all_end = get_time_sync()

    print(f'{config.format()}\n')

    for result in results:
        print(format_result(result))
    print(f'总运行时间: {(all_end - all_start) / 60:.2f} min')

    if config['simple']:
        save_result(results, f'./benchmark/result_simple/{config["backbone"]}_{config["pooler"]}.txt', all_end - all_start)
    else:
        save_result(results, f'./benchmark/result/{config["backbone"]}_{config["pooler"]}.txt', all_end - all_start)

if __name__ == '__main__':
    global run_device
    run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = BenchmarkConfig()
    config.hidden_channels = 64
    config.num_layers = 3
    config.backbone = 'gcn'
    config.pooler = 'topk'
    config.graph_norm = True
    config.batch_size = 128
    config.epochs = 100
    config.simple = False
    config.catch_error = False
    config.early_stop = True
    config.seed = None
    config.kfold = 10

    # models = ['sag', 'asap', 'topk', 'pan', 'struct', 'mincut', 'edge', 'mambo_c_att']
    models = ['topk']
    seeds = [0, 114514, 1919810, 77777]
    for model in models:
        config.pooler = model
        config.seed = seeds[0]
        config.apply_random_seed()

        if config.catch_error:
            try:
                run(config)
            except Exception as e:
                print(f'运行 {config["backbone"]}_{config["pooler"]} 时出错: {e}')
        else:
            run(config)
    