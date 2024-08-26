import os
import torch
from glob import glob
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler  


class CelebADataset(Dataset):
    """
    CelebA dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(self, root, name='celebA', split='train', transform=None, conflict_pct=5):
        self.name = name
        self.transform = transform
        self.root = root
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.header_dir = os.path.join(root, self.name)
        self.data_dir = os.path.join(self.header_dir, "celeba", "img_align_celeba")

        print(f"Reading '{os.path.join(self.header_dir, 'metadata_blonde_validation_cluster.csv')}'")
        self.attrs_df = pd.read_csv(os.path.join(self.header_dir, "metadata_blonde_validation_cluster.csv"))
        self.filename_array = self.attrs_df["image_id"].values
        self.split_array = self.attrs_df["split"].values

        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0


        target_idx = self.attr_idx('Blond_Hair')
        self.y_array = self.attrs_df[:, target_idx]

        confounder_idx = self.attr_idx('Male')
        self.confounder_array = self.attrs_df[:, confounder_idx]

        if split == 'train':
            self.split_token = 0
        elif split == 'test':
            self.split_token = 2
        else:
            self.split_token = 1
        mask = self.split_array == self.split_token

        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        self.filename_array = self.filename_array[indices]
        #print(self.y_array)
        self.y_array = torch.tensor(self.y_array[indices]).long()
        self.confounder_array = torch.tensor(self.confounder_array[indices]).long()
        self.indices = indices


    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])])

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, index, img_filename

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

class ISICDataset(Dataset):
    def __init__(self, root, name='isic', split='train', transform=None, conflict_pct=1):
        self.name = name
        self.transform = transform
        self.root = root
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.header_dir = os.path.join('./dataset/isic')
        #self.header_dir = os.path.join('/projects/datashare/dso/skincancer/raw_224')
        self.data_dir = os.path.join("/scratch_shared/leph/isic/multi_groups")

        #print(f"Reading '{os.path.join(self.header_dir, 'raw_subset1000_val.csv')}'")
        #self.attrs_df = pd.read_csv(os.path.join(self.header_dir, "meta_4groups.csv"))
        self.attrs_df = pd.read_csv(os.path.join(self.header_dir, "raw_val_4groups.csv"))
        self.filename_array = self.attrs_df["isic_id"].values
        self.split_array = self.attrs_df["split"].values

        self.attrs_df = self.attrs_df.drop(labels="isic_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0


        target_idx = self.attr_idx('benign_malignant')
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = np.unique(self.y_array).size
        confounder_idx = self.attr_idx('patches')
        self.confounder_array = self.attrs_df[:, confounder_idx]
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()

        if split == 'train':
            self.split_token = 0
        elif split == 'test':
            self.split_token = 2
        else:
            self.split_token = 1
        mask = self.split_array == self.split_token

        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        self.filename_array = self.filename_array[indices]
        self.y_array = torch.tensor(self.y_array[indices]).long()
        self.confounder_array = torch.tensor(self.confounder_array[indices]).long()
        self.indices = indices


    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])])

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")
        #g = self.group_array[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, attr, index, img_filename

### CUB Dataset
class CUBDataset(Dataset):
    def __init__(self, root, name='cub', split='train', transform=None, conflict_pct=1):
        self.name = name
        self.transform = transform
        self.root = root
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.header_dir = os.path.join(root, 'waterbird')
        #self.header_dir = os.path.join('/projects/datashare/dso/skincancer/raw_224')
        self.data_dir = self.header_dir

        self.attrs_df = pd.read_csv(os.path.join(self.header_dir, "metadata.csv"))
        self.filename_array = self.attrs_df["img_filename"].values
        self.split_array = self.attrs_df["split"].values

        self.attrs_df = self.attrs_df.drop(labels="img_filename", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0


        target_idx = self.attr_idx('y')
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = np.unique(self.y_array).size
        confounder_idx = self.attr_idx('place')
        self.confounder_array = self.attrs_df[:, confounder_idx]
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()

        if split == 'train':
            self.split_token = 0
        elif split == 'test':
            self.split_token = 2
        else:
            self.split_token = 1
        mask = self.split_array == self.split_token

        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        self.filename_array = self.filename_array[indices]
        self.y_array = self.y_array.astype(float)
        self.confounder_array = self.confounder_array.astype(float)
        self.group_array = self.group_array.astype(float)

        self.y_array = torch.tensor(self.y_array[indices]).long()
        self.confounder_array = torch.tensor(self.confounder_array[indices]).long()
        self.group_array = torch.tensor(self.group_array[indices]).long()
        self.indices = indices


    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])])

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")
        #g = self.group_array[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, attr, index, img_filename



    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.y_array[index]), int(self.att_array[index])])

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")
        g = self.group_array[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, attr, img_filename

        
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


