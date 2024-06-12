import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class ColoredMNIST(Dataset):
    def __init__(self, basedir, name='coloredmnist', split='train', transform=None, conflict_pct=5):
        self.basedir = basedir
        self.name = name
        self.transform = transform
        self.folder_dir = os.path.join('../DCWP/dataset/ColoredMNIST')
        self.data_dir = os.path.join('../DCWP/dataset/ColoredMNIST', split)
        if split == 'train':
            self.metadata = pd.read_csv(os.path.join('../DCWP/dataset/ColoredMNIST', 'train_data.csv'))
        else:
            self.metadata = pd.read_csv(os.path.join('../DCWP/dataset/ColoredMNIST', 'valid_data.csv'))
        self.filename_array = self.metadata["idx"].values
        self.y_array = self.metadata["y"].values
        self.att_array = self.metadata["bg"].values
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.att_array).size
        self.group_array = (self.y_array * self.n_places + self.att_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        
        
    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        y = self.y_array[index]
        p = self.att_array[index]

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")
        g = self.group_array[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, y, g, p, img_filename
        
        
class SkinCancerDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        #metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        #metadata_df = pd.read_csv("./data/meta_4groups.csv")
        metadata_df = pd.read_csv("/home/leph/data/isic/raw_val_4groups.csv")
        #metadata_df = pd.read_csv(os.path.join(basedir, "meta_raw_224.csv"))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['benign_malignant'].values
        self.p_array = self.metadata_df['patches'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['patches'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['isic_id'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img_path = os.path.join(self.basedir,'isic/multi_groups', self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y, g, p, idx
#self.filename_array[idx]


def get_transform(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333)),
                #interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(data, train, **kwargs):
    if not train: # Validation or testing
        shuffle = False
        sampler = None
    else: # Training
        shuffle = True
        sampler = None
    print('data len: ',len(data))
    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader

def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    # group_id = y_id * n_places + place_id
    # y_id = group_id // n_places
    # place_id = group_id % n_places
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')
    if val_data is not None:
        logger.write(f'Validation Data (total {len(val_data)})\n')
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')

