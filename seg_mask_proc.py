import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.mydataset import FeatureFusionDataset




if __name__ == '__main__':
    dataset_folder = '/home/wenhuanyao/Dataset/cityscapes/'
    use = 'train'
    mydataset = FeatureFusionDataset(dataset_folder, use=use,if_sp=False, if_seg=True)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)
    output_folder = dataset_folder + use
    
    for i, data in tqdm(enumerate(mydataset)):
        img = data['seg_gt']
        pass