import sys
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


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


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    #preds = (logits>0).int()
    #print('preds',preds)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def write_dict_to_tb(writer, dict, prefix, step):
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)


def get_results(acc_groups, get_yp_func):
    groups = acc_groups.keys()
    results = {
            f"accuracy_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
            for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    results.update({"worst_accuracy" : min(results.values())})
    return results


def evaluate(model, loader, get_yp_func):
    model.eval()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(loader.dataset.n_groups)}
    with torch.no_grad():
        for x, y, g, p, n in tqdm.tqdm(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            #print('logits',logits)
            #y = y.unsqueeze(1)
            #y = y.float()
            update_dict(acc_groups, y, g, logits)
    model.train()
    return get_results(acc_groups, get_yp_func)

