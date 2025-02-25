"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import io
import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile
import imageio
import argparse
import warnings
from pathlib import Path
from itertools import chain
import pickle

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from data.transforms import num_classes, n_groups
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)

def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float))

    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()

class ValidLogger(object):
    phase_token = ['ERM', 'prune', 'retrain', 'ratio']

    def __init__(self, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.fname = fname
        self.log = {
            'ERM': [],
            'prune': [],
            'retrain': [],
            'ratio': [], # Ratio of survived weights during pruning
            'layerwise_ratio': [], # Layerwise ratio of survived weights. Dictionary will be saved
            'groupwise_acc': []
        }

    def append(self, val, which='ERM'):
        self.log[which].append(val)

    def save(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.log, f)
            print(f'saved validation log in {self.fname}')

    def load(self):
        with open(self.fname, 'rb') as f:
            log = pickle.load(f)
            return log

def moving_average_param(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename, denormalize=False):
    if denormalize: x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def plot_embedding(X, label, save_path):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    num_color = np.max(label) + 1
    cmap = plt.cm.get_cmap('rainbow', num_color)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    plt.scatter(X[:, 0], X[:, 1], c=label, cmap='rainbow')

    plt.xticks([]), plt.yticks([])
    #legend = ['source domain {}'.format(i+1) for i in range(min(d), max(d))]
    #legend[-1] = ['target domain']
    #plt.legend(legend)

    fig.savefig(save_path)
    plt.close('all')

def update_dict(acc_groups, y, p, logit):
    #preds = torch.argmax(logits, axis=1)
    preds = logit.data.max(1, keepdim=True)[1].squeeze(1)
    #print('preds',preds)
    correct_batch = (preds == y)
    g = (y*2 + p)
    g_cpu = g.cpu().numpy()
    for g_val in np.unique(g_cpu):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)
def get_results(acc_groups):
    groups = acc_groups.keys()
    results = {
            f"accuracy_{g}": acc_groups[g].avg
            for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    #print('all total', [acc_groups[g].count for g in groups])
    #print('all correct', [acc_groups[g].sum for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    results.update({"worst_accuracy" : min(results.values())})
    return results

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def evaluate_gr(model, fetcher, args):
    #model.eval()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(n_groups[args.data])}
    iterator = enumerate(fetcher)
    for index, (_, data, attr, fname) in iterator:
        y = attr[:, 0].to(device)
        p = attr[:, 1].to(device)
        #g = y*2 + p
        data = data.to(device)

        with torch.no_grad():
            logit = model(data)
            pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
            #correct = (pred == y).long()
            #print('logits',logits)
            y = y.unsqueeze(1)
            y = y.float()
            update_dict(acc_groups, y, p, logit)
    #model.train()
    return get_results(acc_groups)