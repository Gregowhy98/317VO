from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os

def list_files(directory):
    filenames = os.listdir(directory)
    file_paths = [os.path.join(directory, filename) for filename in filenames]
    return file_paths

# TODO
class FeatureFusionTransform(object):
    pass

class FeatureFusionDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None, if_sp=False, raw_seg=None, weighted_seg=None):
        if use not in ['train', 'val', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, val, test]')
        self.dataset_folder = os.path.join(dataset_folder, use)
        self.transform = transform
        self.if_sp = if_sp
        self.raw_seg = raw_seg
        self.weighted_seg = weighted_seg
        
        # load raw image files
        raw_img_folder = os.path.join(self.dataset_folder, 'raw_img')
        self.raw_img_files = list_files(raw_img_folder)
        self.raw_img_files.sort()
        self.N = len(self.raw_img_files)
        
        # load sp gt files
        if self.if_sp:
            sp_gt_folder = os.path.join(self.dataset_folder, 'sp')
            self.sp_gt_files = list_files(sp_gt_folder)
            self.sp_gt_files.sort()
            
            if len(self.raw_img_files) != len(self.sp_gt_files):
                raise ValueError('The number of images and sp gt files do not match')

        # load raw seg files
        if self.raw_seg:
            raw_seg_folder = os.path.join(self.dataset_folder, 'raw_seg')
            self.raw_seg_files = list_files(raw_seg_folder)
            self.raw_seg_files.sort()
            
            if len(self.raw_img_files) != len(self.raw_seg_files):
                raise ValueError('The number of images and raw seg gt files do not match')
        
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
        raw_img = cv2.imread(self.raw_img_files[idx], cv2.IMREAD_UNCHANGED)
        if self.if_sp:
            sp_gt = cv2.imread(self.sp_gt_files[idx], cv2.IMREAD_UNCHANGED)
            sample = {'raw_img': raw_img, 'sp_gt': sp_gt}
            
        if self.raw_seg:
            raw_seg = cv2.imread(self.raw_seg_files[idx], cv2.IMREAD_UNCHANGED)
            raw_seg_path = self.raw_seg_files[idx]
            sample = {'raw_img': raw_img, 'raw_seg': raw_seg, 'raw_seg_path': raw_seg_path}
            
        if self.weighted_seg:
            weighted_seg = cv2.imread(self.weighted_seg_files[idx], cv2.IMREAD_UNCHANGED)
            sample = {'raw_img': raw_img, 'weighted_seg': weighted_seg}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/cityscapes/' 
    mydataset = FeatureFusionDataset(testPath, use='train', if_sp=False, raw_seg=True, weighted_seg=False)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        raw_img = data['raw_img'].numpy()
        seg_gt = data['raw_seg'].numpy()
        pass