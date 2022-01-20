import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold


class ImgDataset(Dataset):
    def __init__(self, transform=None, folder="train"):
        self.transform = transform

        # Load labels
        self.df = pd.read_csv(f"petfinder-pawpularity-score/{folder}.csv")
        self.folder = folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df.iloc[idx]["Id"]
        image = cv2.imread(f"petfinder-pawpularity-score/{self.folder}/{id_}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.folder == "train":
            label = self.df.iloc[idx]['Pawpularity']
        else:
            label = 0

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return id_, image, torch.from_numpy(np.array(label)).float()


class ImgDataset_CE(Dataset):
    def __init__(self, n_classes, transform=None, folder="train"):
        self.transform = transform
        self.n_classes = n_classes

        # Load labels
        self.df = pd.read_csv(f"petfinder-pawpularity-score/{folder}.csv")
        self.folder = folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df.iloc[idx]["Id"]
        image = cv2.imread(f"petfinder-pawpularity-score/{self.folder}/{id_}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.folder == "train":
            label = self.df.iloc[idx]['Pawpularity']
            label_num = torch.from_numpy(np.array(label)).float()
            bins = torch.arange(0, 101, 100 / self.n_classes)
            label_onehot = torch.bucketize(label_num, bins) - 1
            label_onehot = F.one_hot(label_onehot, num_classes=self.n_classes).float()
        else:
            label = 0

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return id_, image, label_onehot, label_num


def get_train_transform(img_size, cj_intesity, cj_p):
    return A.Compose([
        A.SmallestMaxSize(img_size),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=cj_intesity, contrast=cj_intesity, saturation=cj_intesity, hue=cj_intesity, p=cj_p),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0),
        ToTensorV2()
    ])


def get_val_transform(img_size):
    return A.Compose([
        A.SmallestMaxSize(img_size),
        A.CenterCrop(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0),
        ToTensorV2()
    ])


def validation(n_bins, n_splits, seed):
    df = pd.read_csv(f"petfinder-pawpularity-score/train.csv")
    labels = pd.cut(np.array(df["Pawpularity"]), bins=n_bins, labels=False)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = splitter.split(labels, labels)
    return splits
