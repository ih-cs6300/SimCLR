from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pandas as pd
from torch.utils.data import Dataset
import torch
import os

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.5)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]) # for cifar-10
    transforms.Normalize([0.38334954 0.3791455 0.3679523], [0.13129857 0.12903619 0.14591834])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]) # for cifar-10
    transforms.Normalize([0.38334954 0.3791455 0.3679523], [0.13129857 0.12903619 0.14591834])])
    
    


class clevrer_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clevrer_df = df
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["cube", "cylinder", "sphere"]
        self.target_transform = None
        self.targets = list(map(lambda x: self.classes.index(x), self.clevrer_df.iloc[:, -1].tolist()))

    def __len__(self):
        return len(self.clevrer_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.clevrer_df.iloc[idx, 0])
        img = Image.open(img_name)
        target = self.targets[idx]

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)



        return pos_1, pos_2, target

