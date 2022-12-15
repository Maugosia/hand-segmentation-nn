from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as transforms
import torch


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
        if(w > h):
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
        label_one_hot = torch.nn.functional.one_hot(label.to(torch.int64), 2).transpose(1, 3).squeeze().permute(0, 2, 1)

        return image, label_one_hot
