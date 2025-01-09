from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

def list_files(directory):
    filenames = os.listdir(directory)
    file_paths = [os.path.join(directory, filename) for filename in filenames]
    return file_paths

class FeatureFusionDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None, if_sp=False, weighted_seg=False):
        if use not in ['train', 'val', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, val, test]')
        self.dataset_folder = os.path.join(dataset_folder, use)
        self.transform = transform
        self.if_sp = if_sp
        self.weighted_seg = weighted_seg
        
        # load raw image files
        raw_img_folder = os.path.join(self.dataset_folder, 'raw_img')
        self.raw_img_files = list_files(raw_img_folder)
        self.raw_img_files.sort()
        self.N = len(self.raw_img_files)
        
        # load weighted seg files
        if self.weighted_seg:
            weighted_seg_folder = os.path.join(self.dataset_folder, 'weighted_seg')
            self.weighted_seg_files = list_files(weighted_seg_folder)
            self.weighted_seg_files.sort()
            
            if len(self.raw_img_files) != len(self.weighted_seg_files):
                raise ValueError('The number of images and weighted seg gt files do not match')
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        # raw img 3 channels, 0-255, HxWxC, rgb
        raw_img = cv2.imread(self.raw_img_files[idx], cv2.COLOR_BGR2RGB)
        
        # for mynet input 0-1, rgb, 3x320x640
        img_color = torch.from_numpy(raw_img).permute(2, 0, 1).float()
        
        if self.if_sp and self.weighted_seg:
            # for superpoint input 0-1, gray, 1xHxW
            img_gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
            weighted_seg = cv2.imread(self.weighted_seg_files[idx], cv2.IMREAD_UNCHANGED)
            return {'img_color': img_color, 'seg_gt': weighted_seg, 'img_gray':img_gray}
        else:
            return {'img_color': img_color} 
    
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/cityscapes/' 
    tforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 640)),
        transforms.ToTensor()
        ])
    mydataset = FeatureFusionDataset(testPath, use='train', if_sp=False, raw_seg=False, weighted_seg=True, transform=None)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        raw_img = data['raw_img'].numpy()
        seg_gt = data['raw_seg'].numpy()
        pass