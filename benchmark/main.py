import torch
import context
import numpy as np
# from time import perf_counter
from constant import DATASET_PATH
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import Pooler, Classifier
from rich.progress import track
from perf_counter import get_time_sync, measure_time

# proteins_dataset = TUDataset(root=DATASET_PATH, name='PROTEINS')

def train_model(pooler, classifier, train_loader, optimizer, criterion, device):
    pooler.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0
    
    all = len(train_loader)
    cnt = 0
    for data in track(train_loader, description=f"Run train batch: {all}", disable=True):
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

def run(dataset, config):
    stamp_start = get_time_sync()
    # 加载数据集

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(f"设备: {device}")
    
    # 数据集信息
    print(f"数据集: {dataset}")
    # print(f"类别数: {dataset.num_classes}")
    # print(f"节点特征维度: {dataset.num_node_features}")
    
    # 数据集分割
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 数据加载器
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型参数
    input_dim = dataset.num_node_features
    hidden_dim = 64
    # ratio = 0.8  # TopK池化比例
    num_layers = 3
    num_classes = dataset.num_classes
    
    # 创建模型
    pooler = Pooler(input_dim, hidden_dim, num_layers=num_layers, pool_type=config['pooler'], gnn_type=config['backbone'], layer_norm=config['graph_norm']).to(device)
    classifier = Classifier(pooler.get_out_dim(), pooler.get_out_dim(), num_classes).to(device)
    
    # print(f"模型参数数量: {sum(p.numel() for p in pooler.parameters()) + sum(p.numel() for p in classifier.parameters())}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(list(pooler.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    epochs = 100
    best_test_acc = 0

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    
    no_increased_times = 0
    no_record_epoch_num = 20
    early_stop_epoch_num = 30

    for epoch in track(range(epochs), description=f'训练 {dataset}'):
        train_loss, train_acc = train_model(pooler, classifier, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_model(pooler, classifier, test_loader, criterion, device)

        if epoch > no_record_epoch_num:
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
        
        if test_acc > best_test_acc and epoch > no_record_epoch_num:
            best_test_acc = test_acc
        
        if config['early_stop'] and epoch > no_record_epoch_num:
            if test_acc < best_test_acc:
                no_increased_times += 1
            else:
                no_increased_times = 0
            if no_increased_times >= early_stop_epoch_num:
                print(f'Early stop at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    stamp_end = get_time_sync()

    train_acc_mean = np.mean(train_acc_list)
    train_acc_std = np.std(train_acc_list)
    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)
    
    print('训练准确率: {:.2f} ± {:.2f}, 测试准确率 {:.2f} ± {:.2f}'.format(train_acc_mean * 100, train_acc_std * 100, test_acc_mean * 100, test_acc_std * 100))
    print(f'最佳测试准确率: {best_test_acc:.2f}')
    print(f'总运行时间: {(stamp_end - stamp_start) * 1000:.2f} ms\n')
    return (f'{dataset}', test_acc_mean, test_acc_std, best_test_acc)

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
            'DD',
            'PROTEINS',
            'NCI1',
            'NCI109',
            'COX2',
            'IMDB-BINARY',
            'IMDB-MULTI',
            'FRANKENSTEIN',
            'COLLAB',
            'REDDIT-BINARY'
        ]
        for i in range(len(datasets)):
            yield TUDataset(root=DATASET_PATH, name=datasets[i])

def format_result(result):
    return f'数据集: {result[0]}, 测试准确率: {result[1] * 100:.2f}% ± {result[2] * 100:.2f}%, 最佳测试准确率: {result[3] * 100:.2f}%'

def save_result(result, filename, spent_time):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f'backbone: {config["backbone"]}, pooler: {config["pooler"]}, graph_norm: {config["graph_norm"]}, batch_size: {config["batch_size"]}, early_stop: {config["early_stop"]}\n')
        for result in results:
            f.write(f'{format_result(result)}\n')
        f.write(f'总运行时间: {spent_time / 60:.2f} min\n')

if __name__ == '__main__':
    config = {
        'backbone': 'gcn',
        'pooler': 'mambo_att',
        'graph_norm': True,
        'batch_size': 128,
        'simple': True,
        'catch_error': False,
        'early_stop': True
    }

    all_start = get_time_sync()
    results = []
    
    for dataset in track(datasets(config['simple']), description="All"):
        if config['catch_error']:
            try:
                result = run(dataset, config)
                results.append(result)
            except Exception as e:
                print(f'运行 {dataset} 时出错: {e}')
                results.append((dataset, -1, -1))
                continue
        else:
            result = run(dataset, config)
            results.append(result)

    all_end = get_time_sync()

    print(f'backbone: {config["backbone"]}, pooler: {config["pooler"]}, graph_norm: {config["graph_norm"]}, batch_size: {config["batch_size"]}, early_stop: {config["early_stop"]}')
    for result in results:
        print(format_result(result))
    print(f'总运行时间: {(all_end - all_start) / 60:.2f} min')

    if config['simple']:
        save_result(results, f'./benchmark/result_simple/{config["backbone"]}_{config["pooler"]}.txt', all_end - all_start)
    else:
        save_result(results, f'./benchmark/result/{config["backbone"]}_{config["pooler"]}.txt', all_end - all_start)
    