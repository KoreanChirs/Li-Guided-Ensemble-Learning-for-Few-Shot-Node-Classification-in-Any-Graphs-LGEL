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

def create_masks(num_nodes, labels, num_train_per_class=20):
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

def process_dataset(dataset, device):
    data = dataset[0].to(device)
    
    # Create masks if they don't exist
    if not hasattr(data, 'train_mask') or data.train_mask.dim() != 1:
        data.train_mask, data.test_mask = create_masks(data.num_nodes, data.y, num_train_per_class=20)
    
    # Ensure masks are 1-dimensional
    data.train_mask = data.train_mask.squeeze()
    data.test_mask = data.test_mask.squeeze()
    
    # Remove val_mask if it exists
    if hasattr(data, 'val_mask'):
        delattr(data, 'val_mask')
    
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

    # Test models
    gcn_acc = test(gcn, data)
    dnn_acc = test(dnn, data)
    sage_acc = test(sage, data)
    gat_acc = test(gat, data)
    gin_acc = test(gin, data)

    # Compute estimated LI score using GCN predictions
    gcn.eval()
    with torch.no_grad():
        pred = gcn(data.x, data.edge_index).argmax(dim=1).cpu().numpy()
    estimated_li_score = compute_li(pred, data.edge_index.cpu().numpy())

    return estimated_li_score, gcn_acc, dnn_acc, sage_acc, gat_acc, gin_acc

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

    num_attempts = 10
    all_results = []

    for attempt in range(num_attempts):
        print(f"Attempt {attempt + 1}/{num_attempts}")
        
        attempt_results = {
            'gcn': [], 'dnn': [], 'sage': [], 'gat': [], 'gin': [], 'ensemble': []
        }

        for name, dataset in datasets.items():
            try:
                print(f"Processing {name}...")
                estimated_li_score, gcn_acc, dnn_acc, sage_acc, gat_acc, gin_acc = process_dataset(dataset, device)
                
                # Simulate ensemble model
                ensemble_acc = gcn_acc if estimated_li_score > 0.35 else sage_acc

                attempt_results['gcn'].append(gcn_acc)
                attempt_results['dnn'].append(dnn_acc)
                attempt_results['sage'].append(sage_acc)
                attempt_results['gat'].append(gat_acc)
                attempt_results['gin'].append(gin_acc)
                attempt_results['ensemble'].append(ensemble_acc)

                print(f"{name} - Estimated Li: {estimated_li_score:.4f}, GCN: {gcn_acc:.4f}, DNN: {dnn_acc:.4f}, "
                      f"GraphSAGE: {sage_acc:.4f}, GAT: {gat_acc:.4f}, GIN: {gin_acc:.4f}, Ensemble: {ensemble_acc:.4f}")

            except Exception as e:
                print(f"Error processing {name}: {str(e)}")

        # Calculate average accuracies across datasets for this attempt
        avg_results = {model: np.mean(accs) for model, accs in attempt_results.items()}
        all_results.append(avg_results)

    # Calculate overall average and standard deviation across attempts
    final_results = {}
    for model in ['gcn', 'dnn', 'sage', 'gat', 'gin', 'ensemble']:
        model_avgs = [result[model] for result in all_results]
        final_results[model] = (np.mean(model_avgs), np.std(model_avgs))

    # Prepare and display results table
    table_data = [
        ["Model", "Average Accuracy", "Standard Deviation"]
    ]
    for model, (avg, std) in final_results.items():
        table_data.append([model.upper(), f"{avg:.4f}", f"{std:.4f}"])

    print("\nFinal Results Table:")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

    # Optionally, save the table to a file
    with open("final_results_table.txt", "w") as f:
        f.write(tabulate(table_data, headers="firstrow", tablefmt="grid"))


if __name__ == "__main__":
    main()