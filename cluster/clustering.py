import torch
import torchvision
from kmeans_pytorch import kmeans
#from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from dataset import CelebADataset, ISICDataset, CUBDataset, SkinCancerDataset, get_loader, get_transform, log_data

from utils_1 import Logger, AverageMeter, set_seed, evaluate, get_y_p
from sklearn.manifold import TSNE

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
parser = argparse.ArgumentParser(description="k-Mean Clustering Feature Extractor")
parser.add_argument(
    "--data_dir", type=str,
    default='/home/leph/data',
    help="Train dataset directory")
parser.add_argument(
    "--output_dir", type=str,
    default="cluster_celeb/",
    help="Output directory")
parser.add_argument(
    "--checkpoints_dir", type=str,
    default=None,
    help="Checkpoints directory")
parser.add_argument("--augment_data", action='store_true', help="Train data augmentation")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_clusters", type=int, default=16)
parser.add_argument("--seed", type=int, default=1)

args = parser.parse_args()

print('Preparing directory %s' % args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    args_json = json.dumps(vars(args))
    f.write(args_json)

set_seed(args.seed)

def get_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def feature_extractor(model, data_loader):
    features, targets = [], []
    groups = []
    ids = []
    for batch in tqdm.tqdm(data_loader):
        x, attr, n, name = batch
        y = attr[:, 0]
        x = x.to(device)
        y = y.to(device)
        n = n.clone().detach()
        embed = get_embed(model, x)
        features.append(embed)
        targets.append(y)
        ids.append(n)

    features = torch.cat(features)
    targets = torch.cat(targets)
    ids = torch.cat(ids)
    return features, targets, ids

def cluster_features(dataset, features, targets, ids, num_clusters):
    N = len(dataset)
    print(N)
    #sorted_target_clusters = torch.zeros(N).long().to(device) + num_clusters * num_classes

    target_clusters = torch.zeros_like(targets) - 1
    cluster_centers = []
    dct = {}

    #features_t = torch.transpose(features, 0, 1)

    cluster_ids, cluster_center = kmeans(X=features, num_clusters=num_clusters, distance='cosine')
    # cluster_ids, cluster_center = kmeans(X=feature_assigns, num_clusters=num_clusters, distance='cosine', tqdm_flag=False, device=0)
    cluster_ids_ = cluster_ids

    cluster_counts = cluster_ids_.bincount().float()
    print(cluster_counts, len(cluster_counts))

    # target_clusters[target_assigns] = cluster_ids_.to(device)
    cluster_centers.append(cluster_center)

    # sorted_target_clusters = target_clusters
    # cluster_centers = torch.cat(cluster_centers,0)
    return cluster_ids_

## Data Loaders
target_resolution = (224,224)
transform = get_transform(target_resolution=target_resolution, train=True, augment_data=False)
trainset = CUBDataset(root= args.data_dir, split = 'train', transform=transform)

loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 1, 'pin_memory': True}
data_loader = get_loader(trainset, train=True, **loader_kwargs)

num_classes = trainset.n_classes
## Load Model
model = torchvision.models.resnet18(pretrained=False)
d = model.fc.in_features
model.fc = torch.nn.Linear(d, 2)

checkpoints_dir = args.checkpoints_dir
checkpoints = torch.load(checkpoints_dir)
model.load_state_dict(checkpoints['classifier'], strict=False)
model.to(device)


## Clustering
model.eval()
with torch.no_grad():
    features, targets, ids = feature_extractor(model, data_loader)
    num_clusters = args.num_clusters
    cluster_ids = cluster_features(trainset, features, targets, ids, num_clusters)

    cluster_counts = cluster_ids.bincount().float()
    print("Cluster counts : {}, len({})".format(cluster_counts, len(cluster_counts)))

dct = {'index': ids, 'label':targets, 'cluster_assign': cluster_ids}
df= pd.DataFrame.from_dict(dct)
df.to_csv('./clusters.csv', index=False)

X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(features)

torch.save(X_embedded, './tsne.pt')
