from typing import List

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.io import read_image

import pandas as pd
import numpy as np
from PIL import Image


class SkinDataset(Dataset):
    def __init__(self, description_file, out_size, transform_image=None,
                 target_transform=None):
        self.images_file = pd.read_csv(description_file)
        self.transform_image = transform_image
        self.target_transform = target_transform
        self.size = out_size
        self.resize = transforms.Resize(self.size)

    def __len__(self):
        return len(self.images_file)

    def __getitem__(self, idx):
        img_path = self.images_file.iloc[idx, 0]
        image = read_image(img_path)
        w = image.size()[1]
        h = image.size()[2]
        if (w > h):
            diff = w - h
            pad = (0, diff, 0, 0)
        else:
            diff = h - w
            pad = (0, 0, 0, diff)
        if self.transform_image:
            image = self.transform_image(image.to(torch.float64))
        image = torch.nn.functional.pad(image, pad, mode='constant', value=1)
        image = self.resize(image)

        label_path = self.images_file.iloc[idx, 1]
        label = Image.open(label_path)
        if self.target_transform:
            label = self.target_transform(label)
        label = torch.nn.functional.pad(label, pad, mode='constant', value=1)
        label = self.resize(label)
        label_one_hot = torch.nn.functional.one_hot(
            label.to(torch.int64), 2).transpose(1, 3).squeeze().permute(0, 2, 1)

        return image, label_one_hot


def preprocess_datasets(datasets_paths: List[str], image_size: int, mean: torch.Tensor, std: torch.Tensor, train_size: float, val_size: float):
    transform_label = transforms.Compose([transforms.ToTensor()])
    transform_image = transforms.Compose([transforms.Normalize(mean, std)])

    train_datasets = []
    test_datasets = []

    for path in datasets_paths:
        dataset = SkinDataset(
            path, image_size, transform_image, transform_label)
        train_lenght = int(np.floor(train_size * len(dataset)))

        train, test = torch.utils.data.random_split(dataset, [train_lenght, len(dataset) - train_lenght],
                                                    generator=torch.Generator().manual_seed(42))

        train_datasets.append(train)
        test_datasets.append(test)

    data_train_val = torch.utils.data.ConcatDataset(train_datasets)
    data_test = torch.utils.data.ConcatDataset(test_datasets)

    val_lenght = int(np.floor(val_size * len(data_train_val)))
    data_train, data_val = torch.utils.data.random_split(data_train_val, [len(data_train_val) - val_lenght, val_lenght],
                                                         generator=torch.Generator().manual_seed(42))

    return data_train, data_val, data_test
