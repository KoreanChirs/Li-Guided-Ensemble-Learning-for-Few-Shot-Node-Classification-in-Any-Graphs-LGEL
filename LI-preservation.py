import torch
import torch.nn.functional as F
from torch_geometric.datasets import (
    Planetoid, WikipediaNetwork, Actor, HeterophilousGraphDataset,
    Amazon, Coauthor, WebKB, WikipediaNetwork, GNNBenchmarkDataset
)
from torch_geometric.nn import GCN
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
from itertools import cycle

def compute_li(true_labels, pred_labels, edge_index):
    src, dst = edge_index
    num_classes = max(pred_labels.max(), true_labels.max()) + 1
    joint_counts = np.zeros((num_classes, num_classes))
    for s, d in zip(src, dst):
        joint_counts[pred_labels[s], pred_labels[d]] += 1
    
    joint_p = joint_counts / joint_counts.sum()
    p_src = joint_p.sum(axis=1)
    
    mi = mutual_info_score(None, None, contingency=joint_counts)
    h_src = entropy(p_src)
    
    return mi / h_src if h_src != 0 else 0

def train_gcn(data, train_mask):
    num_classes = data.y.max().item() + 1  # Infer number of classes from labels
    model = GCN(data.num_features, 16, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        print(f"{epoch}")
    
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    
    return pred

def run_experiment(data, num_labeled, num_runs=10):
    lis = []
    for _ in range(num_runs):
        perm = torch.randperm(data.num_nodes)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[perm[:num_labeled]] = True
        
        pred_labels = train_gcn(data, train_mask)
        labels = pred_labels.numpy()
        labels[train_mask] = data.y[train_mask].numpy()
        
        li = compute_li(data.y.numpy(), labels, data.edge_index.numpy())
        lis.append(li)
    
    return np.mean(lis)

# Load datasets
# Load datasets
datasets = {
    # Citation networks
    'Cora': Planetoid(root='/tmp/Cora', name='Cora'),
    'CiteSeer': Planetoid(root='/tmp/CiteSeer', name='CiteSeer'),
    'PubMed': Planetoid(root='/tmp/PubMed', name='PubMed'),

    # Coauthor networks (takes long time)
    #'CS': Coauthor(root='/tmp/Coauthor', name='CS'),
    #'Physics': Coauthor(root='/tmp/Coauthor', name='Physics'),

    # Amazon networks (takes long time)
    #'Computers': Amazon(root='/tmp/Amazon', name='Computers'),
    #'Photo': Amazon(root='/tmp/Amazon', name='Photo'),

    # Wikipedia networks
    'Chameleon': WikipediaNetwork(root='/tmp/WikipediaNetwork', name='chameleon'),
    'Squirrel': WikipediaNetwork(root='/tmp/WikipediaNetwork', name='squirrel'),

    # WebKB datasets
    'Cornell': WebKB(root='/tmp/WebKB', name='Cornell'),
    'Texas': WebKB(root='/tmp/WebKB', name='Texas'),
    'Wisconsin': WebKB(root='/tmp/WebKB', name='Wisconsin'),

    # Actor dataset
    'Actor': Actor(root='/tmp/Actor'),

    # Heterophilous datasets
    'Questions': HeterophilousGraphDataset(root='/tmp/HeterophilousGraphDataset', name='questions'),
    #'Roman-empire': HeterophilousGraphDataset(root='/tmp/roman-empire', name='roman-empire'),

    # GNN Benchmark datasets
    #'CLUSTER': GNNBenchmarkDataset(root='/tmp/GNNBenchmark', name='CLUSTER'),
    #'PATTERN': GNNBenchmarkDataset(root='/tmp/GNNBenchmark', name='PATTERN'),

    # You can add more datasets here as needed
}

# Number of labeled nodes to test
labeled_nodes = [5, 10, 15, 20, 25, 30]

results = {dataset_name: [] for dataset_name in datasets.keys()}

for dataset_name, dataset in datasets.items():
    data = dataset[0]
    print(f"Processing {dataset_name}...")
    
    for num_labeled in labeled_nodes:
        avg_li = run_experiment(data, num_labeled)
        results[dataset_name].append(avg_li)
    
    # Add LI for 100% labeled nodes
    results[dataset_name].append(compute_li(data.y.numpy(), data.y.numpy(), data.edge_index.numpy()))

# Plotting
plt.figure(figsize=(15, 10))

# Generate colors dynamically based on the number of datasets
num_datasets = len(results)
colors = plt.cm.tab20(np.linspace(0, 1, num_datasets))

markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', '+', 'x', '1', '2', '3', '4']

# Iterate through datasets to plot their LI values
for idx, (dataset_name, lis) in enumerate(results.items()):
    plt.plot(range(len(labeled_nodes) + 1), lis, label=dataset_name, 
             color=colors[idx], marker=markers[idx % len(markers)])

# Set x-axis ticks and labels
xlabels = [str(x) for x in labeled_nodes] + ['100%']
plt.xticks(ticks=range(len(xlabels)), labels=xlabels)

plt.xlabel('Number of Labeled Nodes')
plt.ylabel('Label Informativeness (LI)')
plt.title('Label Informativeness vs. Number of Labeled Nodes for Various Datasets')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)

# Add vertical lines for specific numbers of labeled nodes
for i in range(len(labeled_nodes)):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
plt.show()