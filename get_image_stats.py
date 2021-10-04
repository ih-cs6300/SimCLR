import torch
import numpy
import torchvision.datasets as datasets
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import utils

#cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# read in csv
all_df = pd.read_csv('/home/iharmon1/data/clevrer_shapes/shapes/all_ds.csv')
train_df, test_df = train_test_split(all_df, test_size = 0.30, random_state = 42)
trainset = utils.clevrer_dataset(train_df, root_dir='/home/iharmon1/data/clevrer_shapes/shapes', train=True, transform=transforms.ToTensor())


imgs = [item[0] for item in trainset] # item[0] and item[1] are image and its label
imgs = torch.stack(imgs, dim=0).numpy()

# calculate mean over each channel (r,g,b)
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(mean_r,mean_g,mean_b)

# calculate std over each channel (r,g,b)
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(std_r,std_g,std_b)
