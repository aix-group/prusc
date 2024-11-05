import torch
import torchvision
from kmeans_pytorch import kmeans
#from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import os
import tqdm
import argparse
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as T
from torch import nn, optim, autograd

import random
from networks import MNIST_CNN
#from optimizer import SGD
#from utils_1 import update_dict, get_results, write_dict_to_tb
#from models import ConvModel

from sklearn.manifold import TSNE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Representation Clustering')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument(
    "--output_dir", type=str,
    default="logs/",
    help="Output directory")
parser.add_argument(
    "--train_data", type=str,
    default="custom_data/train15_noise25.pt",
    help="Custom training dataset with spurious ratio and noise level")
flags = parser.parse_args()


## Loading saved training
envs = [
  torch.load(flags.train_data)
]

train = torch.utils.data.TensorDataset(envs[0]['idx'], envs[0]['images'], envs[0]['labels'], envs[0]['ground_truth'], envs[0]['colors'])
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)


def get_embed(m, x):
    x = m.network.conv1(x)
    x = m.network.bn1(x)
    x = m.network.relu(x)
    x = m.network.maxpool(x)

    x = m.network.layer1(x)
    x = m.network.layer2(x)
    x = m.network.layer3(x)
    x = m.network.layer4(x)

    x = m.network.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def get_embed_mnist(model, x):
    x = model.conv1(x)
    x = F.relu(x)
    x = model.bn0(x)

    x = model.conv2(x)
    x = F.relu(x)
    x = model.bn1(x)

    x = model.conv3(x)
    x = F.relu(x)
    x = model.bn2(x)

    x = model.conv4(x)
    x = F.relu(x)
    x = model.bn3(x)

    x = model.avgpool(x)
    x = x.view(len(x), -1)
    return x

def feature_extractor(model, data_loader):
    features, targets = [], []
    grounds = []
    ids = []
    colors = []
    for batch in tqdm.tqdm(data_loader):
        #x, y, g, p, n = batch
        i, x, y, g, c = batch
        i = i.to(device)
        x = x.to(device)
        y = y.to(device)
        g = g.to(device)
        c = c.to(device)
        embed = get_embed_mnist(model, x)
        features.append(embed)
        targets.append(y)
        grounds.append(g)
        ids.append(i)
        colors.append(c)

    features = torch.cat(features)
    targets = torch.cat(targets)
    grounds = torch.cat(grounds)
    ids = torch.cat(ids)
    colors = torch.cat(colors)
    return features, targets, ids, grounds, colors

def cluster_features(dataset, features, targets, ids, num_clusters):
    N = len(dataset)
    target_clusters = torch.zeros_like(targets) - 1
    cluster_centers = []
    dct = {}
    cluster_ids, cluster_center = kmeans(X=features, num_clusters=num_clusters, distance='cosine')
    cluster_ids_ = cluster_ids

    cluster_counts = cluster_ids_.bincount().float()
    print(cluster_counts, len(cluster_counts))
    cluster_centers.append(cluster_center)

    return cluster_ids_


model = MNIST_CNN(2)
checkpoint = torch.load(flags.ckpt)
model.load_state_dict(checkpoint)
model.to(device)

## Clustering
model.eval()
with torch.no_grad():
    features, targets, ids, g, c = feature_extractor(model, train_loader)
    num_clusters = 8
    cluster_ids = cluster_features(train, features, targets, ids, num_clusters)

    cluster_counts = cluster_ids.bincount().float()
    print("Cluster counts : {}, len({})".format(cluster_counts, len(cluster_counts)))

dct = {'index': ids.cpu().numpy().flatten(), 'label_noise':targets.cpu().numpy().flatten(), 'grounds':g.cpu().numpy().flatten(), 'colors':c.cpu().numpy().flatten(), 'cluster_assign': cluster_ids.cpu().numpy().flatten()}
df= pd.DataFrame.from_dict(dct)
df.to_csv(os.path.join(flags.output_dir, 'rep_clustering.csv'), index=False)

## for visualisation
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(features.cpu().numpy())

