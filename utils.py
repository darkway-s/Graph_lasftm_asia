import networkx as nx
import pandas as pd
import json
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_graph():
  edges_data = pd.read_csv('lastfm_asia_edges.csv')

  labels_data = pd.read_csv('lastfm_asia_target.csv')
  # with open('/Users/xielinni/Desktop/MSBD5008/project/dataset/lastfm_asia_features.json', 'r') as f:
  with open('lastfm_asia_features.json', 'r') as f:
      features_data = json.load(f)

  graph = nx.Graph()

  # add nodes to the graph
  for node_id in labels_data['id']:
      graph.add_node(node_id)

  # add labels to the graph
  for _, row in labels_data.iterrows():
      node_id = row['id']
      label = row['target']
      graph.nodes[node_id]['target'] = label

  # add edges to the graph
  for _, edge in edges_data.iterrows():
      source = edge['node_1']
      target = edge['node_2']
      graph.add_edge(source, target)

  # add features to the graph
  for node_id, features in features_data.items():
      node_id = int(node_id)
      graph.nodes[node_id]['features'] = features

  return graph, edges_data, labels_data, features_data

def get_mask(graph):
  g = dgl.from_networkx(graph)

  train_mask = np.zeros(len(graph), dtype=bool)
  train_mask[:int(len(train_mask)*0.6)] = True
  val_mask = np.zeros(len(graph), dtype=bool)
  val_mask[int(len(train_mask)*0.6):int(len(val_mask)*0.8)] = True
  test_mask = np.zeros(len(graph), dtype=bool)
  test_mask[int(len(test_mask)*0.8):] = True
  # same_indices = np.where(test_mask == val_mask)[0]
  # print(same_indices)

  return g, train_mask, val_mask, test_mask

def get_labels(labels_data):
  labels = torch.tensor(labels_data['target'])
  unique_values, counts = np.unique(labels, return_counts=True)
  num_classes = len(unique_values)

  return  labels, num_classes

def get_naive_feat(graph):
  # naive feat: 1 for all nodes
  naive_feat = torch.unsqueeze(torch.tensor(np.ones(len(graph)), dtype=torch.float), dim=1)
  return naive_feat

def get_property_feat(graph):
  # property feat: degree, clustering coefficient, eigenvector centrality for each node
  degree_sequence = [graph.degree(node) for node in graph.nodes()]
  clustering_coefficients = nx.clustering(graph)
  clustering_coefficients = [clustering_coefficients[n] for n in graph.nodes()]
  eigenvector_centrality = nx.eigenvector_centrality(graph)
  eigenvector_centrality = [eigenvector_centrality[n] for n in graph.nodes()]
  property_feat = torch.tensor([
      [degree_sequence[i], clustering_coefficients[i], eigenvector_centrality[i]]
      for i in range(len(graph))
  ])
  
  return property_feat

def get_real_feat(graph, features_data):
  #real features
  max_feature = max(max(lst) for lst in features_data.values() if lst)

  feat = torch.zeros(len(graph), max_feature+1, dtype=torch.bool)
  for i, lst in enumerate(features_data.values()):
    if lst:
      feat[i, lst] = True
  cols_to_remove = torch.all(feat == False, dim=0)
  cols_to_keep = ~cols_to_remove
  feat = feat[:, cols_to_keep]
  feat = feat.float()
  
  return feat
  
  

