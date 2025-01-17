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
from torch.optim import Adam 

df_edges_data = pd.read_csv("edge.csv")
relation_embeddings = np.load("emb_transr_R_8.npy")
print(relation_embeddings)
relation_embeddings = np.concatenate((np.zeros((1, relation_embeddings.shape[1])), relation_embeddings), axis=0)

print(relation_embeddings.shape)

edge_feat_names = ['r2_score', 'r5_score', 'r7_score', 'r1_score', 'r3_score', 'r6_score', 'r4_score', 'r8_score']

relation_map = torch.tensor(np.where(np.array(df_edges_data[edge_feat_names]) > 0, 1, 0))

relation_map_coords = torch.nonzero(relation_map)
relation_map[relation_map_coords[:, 0], relation_map_coords[:, 1]] = relation_map_coords[:, 1] + 1

weight_map = np.array(df_edges_data[edge_feat_names])

print(relation_map.shape)

print('finished')
edge_features = []

for row, weight in zip(relation_map, weight_map):
    edge_feature = np.multiply(relation_embeddings[row], weight[:, np.newaxis])
    edge_feature = edge_feature.flatten()
    edge_features.append(edge_feature)
            

edge_features = np.array(edge_features)
print(edge_features.shape)


# Save new edge features to a CSV file
new_edge_features_df = pd.DataFrame(edge_features, columns=['edge_feat' + str(i) for i in range(edge_features.shape[1])])
new_edge_features_df['src'] = torch.from_numpy(df_edges_data['src'].to_numpy())
new_edge_features_df['dst'] = torch.from_numpy(df_edges_data['dst'].to_numpy())
new_edge_features_df.to_csv("KG_Transr_64_concat_upload_df_edges_data.csv", index=False)
print('finished')
