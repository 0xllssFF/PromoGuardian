### Define Graph
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import DataLoader
import pandas as pd
import numpy as np
import networkx as nx

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from sklearn.metrics import precision_score, recall_score, accuracy_score

class YouxuanDataset(DGLDataset):
    def __init__(self, df_nodes_data, df_edges_data, node_feat_size, edge_feat_names, mode = 'train'):
        self.df_nodes_data = df_nodes_data
        self.df_edges_data = df_edges_data
        self.node_feat_size = node_feat_size
        self.edge_feat_names = edge_feat_names
        self.mode = mode
        super().__init__(name="youxuan")

        
    def process(self): 
        self.df_edges_data[self.edge_feat_names] = self.df_edges_data[self.edge_feat_names].astype(float)
        self.df_nodes_data['label'] = self.df_nodes_data['label'].replace(-1, 2)

        id = torch.from_numpy(self.df_nodes_data['id'].to_numpy())
        
        node_features = torch.ones(self.df_nodes_data.to_numpy().shape[0], self.node_feat_size, dtype=torch.float32)

        node_labels = torch.from_numpy(self.df_nodes_data['label'].astype('category').cat.codes.to_numpy())


        edge_features = torch.from_numpy(self.df_edges_data[self.edge_feat_names].to_numpy().astype(np.float32))
        
        edges_src = torch.from_numpy(self.df_edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(self.df_edges_data['dst'].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=self.df_nodes_data.shape[0]
        )
        self.graph.nodes[id].data["feat"] = node_features
        self.graph.nodes[id].data['id'] = id
        self.graph.nodes[id].data["label"] = node_labels
        self.graph.edata["weight"] = edge_features

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1
    
# saved version
class AttentionAggregationGNN(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, attention_size, transformed_node_feature_size, num_attention_layers):
        super(AttentionAggregationGNN, self).__init__()
        self.num_attention_layers = num_attention_layers
        self.W_node = nn.Linear(node_feature_size, transformed_node_feature_size)
        self.node_proj = nn.ModuleList([nn.Linear(transformed_node_feature_size, attention_size) for _ in range(num_attention_layers)])
        self.edge_proj = nn.Linear(edge_feature_size, attention_size)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, g, node_features, edge_features):
        # Project node and edge features
        transformed_node_feature = self.W_node(node_features)
        node_reprs = [node_proj(transformed_node_feature) for node_proj in self.node_proj]
        edge_repr = self.edge_proj(edge_features)
        
        attention_scores = []
        for i in range(self.num_attention_layers):
            # Use DGL's u_mul_v to compute the attention scores
            g.ndata['hv'] = node_reprs[i]
            g.edata['ev'] = edge_repr
            g.apply_edges(lambda edges: {'h': (edges.src['hv'] * edges.data['ev']).sum(1)})
            # Convert the computed sum to attention scores
            g.edata['attention'] = self.leaky_relu(g.edata['h'])
            # Apply softmax to get attention weights
            g.edata['attention'] = F.softmax(g.edata['attention'], dim=0)
            attention_scores.append(g.edata['attention'])
        
        # Aggregate neighbor features using the attention weights
        g.ndata['h'] = transformed_node_feature
        g.edata['a'] = torch.stack(attention_scores, dim=1).mean(dim = 1).view(-1, 1)
        g.update_all(message_func=self.src_mul_edge_message_func,
                     reduce_func=dgl.function.sum('message', 'h_agg'))
        updated_node_features = g.ndata['h_agg']
        return updated_node_features, g.edata['a']
    
    @staticmethod
    def src_mul_edge_message_func(edges):
        return {'message': edges.src['h'] * edges.data['a']}


    
class AE(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(AE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, encoding_dim)
        
        # Decoder
        self.fc3 = nn.Linear(encoding_dim, 32)
        self.fc4 = nn.Linear(32, input_size)
        
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        encoded_x = self.fc2(h1)
        return encoded_x

    def decode(self, z):
        h4 = F.relu(self.fc3(z))
        reconstructed_x = self.fc4(h4)
        return reconstructed_x

    def forward(self, x):
        encoded_x = self.encode(x)
        reconstructed_x = self.decode(encoded_x)
        return reconstructed_x
    
# Parameters
node_feature_size = 52  # Example feature size for each node
transformed_node_feature_size = 52 # Example transformed feature size for each node
edge_feature_size = 64  # Example feature size for each edge
attention_size = 8     # Size of the intermediate attention representation
encoding_dim = 16        # Size of the encoded representation
attention_layer_num = 3
learning_rate = 1e-4
w_d = 1e-8


# Instantiate models
gnn_layer = AttentionAggregationGNN(node_feature_size, edge_feature_size, attention_size, transformed_node_feature_size, attention_layer_num).to('cpu')
vae = AE(transformed_node_feature_size, encoding_dim).to('cpu')


# Instantiate the Adam optimizer for both GNN and VAE parameters
optimizer = torch.optim.Adam(
    list(gnn_layer.parameters()) + list(vae.parameters()),
    lr=learning_rate,
    weight_decay=w_d
)

### Test
### Load Test Dataset
test_df_nodes_data = pd.read_csv("node.csv")
test_df_edges_data = pd.read_csv("KG_Transr_64_concat_upload_df_edges_data.csv")

test_df_nodes_data = test_df_nodes_data.reset_index(drop=True)
test_df_edges_data = test_df_edges_data.reset_index(drop=True)

print(test_df_edges_data.columns.values)
print(test_df_nodes_data.columns.values)

test_edge_feat_names = ['edge_feat' + str(i) for i in range(edge_feature_size)]
test_dataset = YouxuanDataset(test_df_nodes_data, test_df_edges_data, node_feature_size, test_edge_feat_names, 'test')
test_g = test_dataset[0]
test_g = dgl.add_reverse_edges(test_g, copy_ndata=True, copy_edata=True)
test_g = test_g.to('cpu')
print(test_g)

### Load Model
checkpoint = torch.load('model_checkpoint_weighted_TransR.pth',map_location=torch.device('cpu'))
# checkpoint = torch.load('./ablation/model_checkpoint_r.pth')

gnn_layer.load_state_dict(checkpoint['gnn_state_dict'])
vae.load_state_dict(checkpoint['vae_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
min_loss = checkpoint['min_loss']


updated_node_features, _ = gnn_layer(test_g, test_g.ndata['feat'], test_g.edata['weight'])
reconstructed_node_features = vae(updated_node_features)

print(test_g.edata['a'].shape,test_g.edata['a'].type())
attentionvalue = test_g.edata['a'].cpu() 
torch.save(attentionvalue, 'attentionvalue.pt')

reconstruction_loss = torch.nn.MSELoss(reduction='none')(reconstructed_node_features, updated_node_features).sum(dim = 1)
print(reconstructed_node_features.shape)

ano_topk = 0.012
num_nodes = test_g.num_nodes()
test_range = torch.arange(num_nodes)

num_ano = int(num_nodes * ano_topk)
anomaly_score, ano_idx = torch.topk(reconstruction_loss, num_ano)
ano_idx = ano_idx.to('cpu')
anomaly_score = anomaly_score.to('cpu').detach().numpy()
print(len(ano_idx))

print(f"{anomaly_score[-1]:.15f}")


node_anomaly_score = {}

for i, idx in enumerate(ano_idx):
    node_anomaly_score[int(test_g.ndata['id'][idx].to('cpu'))] = float(anomaly_score[i])
    
import json
json.dump(node_anomaly_score, open('node_anomaly_score.json', 'w'))

node_idx2userid = []
for idx in ano_idx:
    node_idx2userid.append(int(test_g.ndata['id'][idx].to('cpu')))

f = open('result-relabel.txt', 'w')
for item in node_idx2userid:
    f.write(str(item) + '\n')
f.close()

initial_nodes = []
with open('result-relabel.txt', 'r') as file:
    for line in file:
        node = line.strip() 
        if node:
            initial_nodes.append(int(node))

edges_df = pd.read_csv('./edge.csv')
edges = edges_df[['src', 'dst']].values.tolist()

weights_tensor = torch.load('attentionvalue.pt')
weights_tensor = weights_tensor.squeeze(1)
weights_list = weights_tensor.tolist()
G = nx.Graph()

for i, edge in enumerate(edges):
    G.add_edge(edge[0], edge[1], weight=weights_list[i])

node_scores = dict.fromkeys(G.nodes, 0)
for node in initial_nodes:
    if node in G.nodes:
        node_scores[node] = 1.0

def bfs_score_allocation(G, initial_nodes, max_distance=3):
    visited = set()
    queue = deque([(node, 0) for node in initial_nodes])

    while queue:
        current_node, distance = queue.popleft()
        if distance > max_distance:
            continue

        visited.add(current_node)
        
        for neighbor in G.neighbors(current_node):
            if neighbor not in visited :
                edge_data = G.get_edge_data(current_node, neighbor)
                w = edge_data['weight']
                node_scores[neighbor] += min(1.0,node_scores[current_node])*w*1e6
                queue.append((neighbor, distance + 1))

bfs_score_allocation(G, initial_nodes, max_distance=3)

high_score_nodes = {node for node, score in node_scores.items() if score >= 0.65}

result_df = pd.DataFrame(list(high_score_nodes), columns=['Node'])
result_df.to_csv('high_score_nodes.csv', index=False)

def calculate_metrics(high_score_path, node_csv_path):
    high_score_df = pd.read_csv(high_score_path)
    predicted_positive_nodes = set(high_score_df['Node'])

    node_df = pd.read_csv(node_csv_path)

    y_true = []
    y_pred = []

    for _, row in node_df.iterrows():
        id = row['id']
        true_label = row['label']

        y_true.append(1 if true_label == 1 else 0)
        y_pred.append(1 if id in predicted_positive_nodes else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, accuracy

high_score_path = 'high_score_nodes.csv'
node_csv_path = 'node.csv'
precision, recall, accuracy = calculate_metrics(high_score_path, node_csv_path)

if precision is not None and recall is not None and accuracy is not None:
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
