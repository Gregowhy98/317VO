import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as DataLoader


from datasets.wireframedataset import WireframeDataset







dataset_folder = '/home/wenhuanyao/Dataset/Wireframe'
myvodataset = WireframeDataset(dataset_folder='datasets/wireframe', use='train')
myvodatasetloader = DataLoader(myvodataset, batch_size=1, shuffle=False)    # false for vo demo

for i, data in enumerate(myvodatasetloader):