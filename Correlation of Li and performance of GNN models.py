import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import (
    Planetoid, WikipediaNetwork, Actor, HeterophilousGraphDataset,
    Amazon, Coauthor, WebKB, GNNBenchmarkDataset
)
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from tabulate import tabulate

# Models
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class DNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def compute_li(true_labels, edge_index):
    src, dst = edge_index
    num_classes = true_labels.max() + 1
    joint_counts = np.zeros((num_classes, num_classes))
    
    for s, d in zip(src, dst):
        joint_counts[true_labels[s], true_labels[d]] += 1
    
    joint_p = joint_counts / joint_counts.sum()
    p_src = joint_p.sum(axis=1)
    
    mi = mutual_info_score(None, None, contingency=joint_counts)
    h_src = entropy(p_src)
    
    return mi / h_src if h_src != 0 else 0

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    if isinstance(model, DNN):
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def test(model, data):
    model.eval()
    if isinstance(model, DNN):
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def create_masks(num_nodes, labels, num_train_per_class=10):
    # Initialize masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Get unique classes
    classes = torch.unique(labels)
    
    # Select train nodes
    for c in classes:
        idx = torch.where(labels == c)[0]
        if len(idx) > num_train_per_class:
            train_idx = idx[torch.randperm(len(idx))[:num_train_per_class]]
        else:
            train_idx = idx
        train_mask[train_idx] = True
    
    # All non-training nodes are used for testing
    test_mask = ~train_mask
    
    return train_mask, test_mask

def process_dataset(dataset, device, num_runs=10):
    data = dataset[0].to(device)
    
    # Create masks if they don't exist
    if not hasattr(data, 'train_mask') or data.train_mask.dim() != 1:
        data.train_mask, data.test_mask = create_masks(data.num_nodes, data.y, num_train_per_class=10)
    
    # Ensure masks are 1-dimensional
    data.train_mask = data.train_mask.squeeze()
    data.test_mask = data.test_mask.squeeze()
    
    # Remove val_mask if it exists
    if hasattr(data, 'val_mask'):
        delattr(data, 'val_mask')
    
    # Compute Li score
    true_labels_np = data.y.cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy()
    li_score = compute_li(true_labels_np, edge_index_np)
    
    # Initialize accumulators for accuracies
    gcn_accs, dnn_accs, sage_accs, gat_accs, gin_accs = [], [], [], [], []
    
    for _ in range(num_runs):
        # Initialize models
        num_classes = data.y.max().item() + 1
        gcn = GCN(data.num_features, 64, num_classes).to(device)
        dnn = DNN(data.num_features, 64, num_classes).to(device)
        sage = GraphSAGE(data.num_features, 64, num_classes).to(device)
        gat = GAT(data.num_features, 8, num_classes).to(device)
        gin = GIN(data.num_features, 64, num_classes).to(device)
        
        # Initialize optimizers
        gcn_optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
        dnn_optimizer = torch.optim.Adam(dnn.parameters(), lr=0.01, weight_decay=5e-4)
        sage_optimizer = torch.optim.Adam(sage.parameters(), lr=0.01, weight_decay=5e-4)
        gat_optimizer = torch.optim.Adam(gat.parameters(), lr=0.005, weight_decay=5e-4)
        gin_optimizer = torch.optim.Adam(gin.parameters(), lr=0.01, weight_decay=5e-4)

        # Train models
        for epoch in range(200):
            train(gcn, gcn_optimizer, data)
            train(dnn, dnn_optimizer, data)
            train(sage, sage_optimizer, data)
            train(gat, gat_optimizer, data)
            train(gin, gin_optimizer, data)

        # Test models and accumulate accuracies
        gcn_accs.append(test(gcn, data))
        dnn_accs.append(test(dnn, data))
        sage_accs.append(test(sage, data))
        gat_accs.append(test(gat, data))
        gin_accs.append(test(gin, data))

    # Calculate average accuracies
    gcn_acc = sum(gcn_accs) / num_runs
    dnn_acc = sum(dnn_accs) / num_runs
    sage_acc = sum(sage_accs) / num_runs
    gat_acc = sum(gat_accs) / num_runs
    gin_acc = sum(gin_accs) / num_runs

    return li_score, gcn_accs, dnn_accs, sage_accs, gat_accs, gin_accs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = {
        'Cora': Planetoid(root='/tmp/Cora', name='Cora'),
        'CiteSeer': Planetoid(root='/tmp/CiteSeer', name='CiteSeer'),
        'PubMed': Planetoid(root='/tmp/PubMed', name='PubMed'),
        'Computers': Amazon(root='/tmp/Amazon', name='Computers'),
        'Photo': Amazon(root='/tmp/Amazon', name='Photo'),
        'Chameleon': WikipediaNetwork(root='/tmp/WikipediaNetwork', name='chameleon'),
        'Squirrel': WikipediaNetwork(root='/tmp/WikipediaNetwork', name='squirrel'),
        'Cornell': WebKB(root='/tmp/WebKB', name='Cornell'),
        'Texas': WebKB(root='/tmp/WebKB', name='Texas'),
        'Wisconsin': WebKB(root='/tmp/WebKB', name='Wisconsin'),
        'Actor': Actor(root='/tmp/Actor'),
        'Questions': HeterophilousGraphDataset(root='/tmp/HeterophilousGraphDataset', name='questions'),
        'Roman-empire': HeterophilousGraphDataset(root='/tmp/roman-empire', name='roman-empire'),
        'CLUSTER': GNNBenchmarkDataset(root='/tmp/GNNBenchmark', name='CLUSTER'),
        'PATTERN': GNNBenchmarkDataset(root='/tmp/GNNBenchmark', name='PATTERN')
    }

    results = {}
    low_li_results = {}
    high_li_results = {}

    for name, dataset in datasets.items():
        try:
            print(f"Processing {name}...")
            li_score, gcn_accs, dnn_accs, sage_accs, gat_accs, gin_accs = process_dataset(dataset, device)
            
            gcn_acc_mean = np.mean(gcn_accs)
            gcn_acc_std = np.std(gcn_accs)
            dnn_acc_mean = np.mean(dnn_accs)
            dnn_acc_std = np.std(dnn_accs)
            sage_acc_mean = np.mean(sage_accs)
            sage_acc_std = np.std(sage_accs)
            gat_acc_mean = np.mean(gat_accs)
            gat_acc_std = np.std(gat_accs)
            gin_acc_mean = np.mean(gin_accs)
            gin_acc_std = np.std(gin_accs)
            
            avg_acc = (gcn_acc_mean + dnn_acc_mean + sage_acc_mean + gat_acc_mean + gin_acc_mean) / 5
            
            results[name] = {
                'li_score': li_score,
                'gcn_acc': (gcn_acc_mean, gcn_acc_std),
                'dnn_acc': (dnn_acc_mean, dnn_acc_std),
                'sage_acc': (sage_acc_mean, sage_acc_std),
                'gat_acc': (gat_acc_mean, gat_acc_std),
                'gin_acc': (gin_acc_mean, gin_acc_std),
                'avg_acc': avg_acc
            }
            
            print(f"{name} - Li: {li_score:.4f}, "
                  f"GCN: {gcn_acc_mean:.4f} ± {gcn_acc_std:.4f}, "
                  f"DNN: {dnn_acc_mean:.4f} ± {dnn_acc_std:.4f}, "
                  f"GraphSAGE: {sage_acc_mean:.4f} ± {sage_acc_std:.4f}, "
                  f"GAT: {gat_acc_mean:.4f} ± {gat_acc_std:.4f}, "
                  f"GIN: {gin_acc_mean:.4f} ± {gin_acc_std:.4f}, "
                  f"Avg: {avg_acc:.4f}")
            
            if li_score < 0.3:
                low_li_results[name] = results[name]
            else:
                high_li_results[name] = results[name]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")

    # Calculate overall average accuracy for Li < 0.3 and Li > 0.3
    if low_li_results:
        low_li_avg = sum(result['avg_acc'] for result in low_li_results.values()) / len(low_li_results)
        print(f"\nAverage accuracy for datasets with Li < 0.3: {low_li_avg:.4f}")
        best_model_low_li = max(
            ('GCN', sum(result['gcn_acc'][0] for result in low_li_results.values()) / len(low_li_results)),
            ('DNN', sum(result['dnn_acc'][0] for result in low_li_results.values()) / len(low_li_results)),
            ('GraphSAGE', sum(result['sage_acc'][0] for result in low_li_results.values()) / len(low_li_results)),
            ('GAT', sum(result['gat_acc'][0] for result in low_li_results.values()) / len(low_li_results)),
            ('GIN', sum(result['gin_acc'][0] for result in low_li_results.values()) / len(low_li_results)),
            key=lambda x: x[1]
        )
        print(f"Best model for Li < 0.3: {best_model_low_li[0]} with average accuracy {best_model_low_li[1]:.4f}")
    else:
        print("\nNo datasets with Li < 0.3")

    if high_li_results:
        high_li_avg = sum(result['avg_acc'] for result in high_li_results.values()) / len(high_li_results)
        print(f"\nAverage accuracy for datasets with Li > 0.3: {high_li_avg:.4f}")
        best_model_high_li = max(
            ('GCN', sum(result['gcn_acc'][0] for result in high_li_results.values()) / len(high_li_results)),
            ('DNN', sum(result['dnn_acc'][0] for result in high_li_results.values()) / len(high_li_results)),
            ('GraphSAGE', sum(result['sage_acc'][0] for result in high_li_results.values()) / len(high_li_results)),
            ('GAT', sum(result['gat_acc'][0] for result in high_li_results.values()) / len(high_li_results)),
            ('GIN', sum(result['gin_acc'][0] for result in high_li_results.values()) / len(high_li_results)),
            key=lambda x: x[1]
        )
        print(f"Best model for Li > 0.3: {best_model_high_li[0]} with average accuracy {best_model_high_li[1]:.4f}")
    else:
        print("\nNo datasets with Li > 0.3")

    # Prepare data for the table
    table_data = []
    for name, result in results.items():
        model_accuracies = {
            'GCN': result['gcn_acc'][0],
            'DNN': result['dnn_acc'][0],
            'GraphSAGE': result['sage_acc'][0],
            'GAT': result['gat_acc'][0],
            'GIN': result['gin_acc'][0]
        }
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_models[0][0]
        second = sorted_models[1][0]

        table_data.append([
            name,
            f"{result['li_score']:.4f}",
            f"{result['gcn_acc'][0]:.4f} ± {result['gcn_acc'][1]:.4f}",
            f"{result['dnn_acc'][0]:.4f} ± {result['dnn_acc'][1]:.4f}",
            f"{result['sage_acc'][0]:.4f} ± {result['sage_acc'][1]:.4f}",
            f"{result['gat_acc'][0]:.4f} ± {result['gat_acc'][1]:.4f}",
            f"{result['gin_acc'][0]:.4f} ± {result['gin_acc'][1]:.4f}",
            f"{result['avg_acc']:.4f}",
            f"{winner} ({model_accuracies[winner]:.4f})",
            f"{second} ({model_accuracies[second]:.4f})"
        ])

    # Sort the table data by Li score
    table_data.sort(key=lambda x: float(x[1]))

    # Create and display the table
    headers = ["Dataset", "Li Score", "GCN", "DNN", "GraphSAGE", "GAT", "GIN", "Avg Acc", "Winner", "2nd Place"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print("\nResults Table:")
    print(table)

    # Optionally, save the table to a file
    with open("appendix_10.txt", "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()