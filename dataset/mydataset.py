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
    def __init__(self, dataset_folder, transform=None, if_sp=False, if_seg=True):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.if_sp = if_sp
        self.if_seg = if_seg
        
        # load raw image files
        raw_img_folder = self.dataset_folder + 'raw/'
        self.raw_img_files = list_files(raw_img_folder)
        self.raw_img_files.sort()
        self.N = len(self.raw_img_files)
        
        # load seg gt files
        if self.if_seg:
            seg_gt_folder = self.dataset_folder + 'seg/'
            self.seg_gt_files = list_files(seg_gt_folder)
            self.seg_gt_files.sort()
            
            if len(self.raw_img_files) != len(self.seg_gt_files):
                raise ValueError('The number of images and seg gt files do not match')
        
        # load sp gt files
        if self.if_sp:
            sp_gt_folder = self.dataset_folder + 'sp/'
            self.sp_gt_files = list_files(sp_gt_folder)
            self.sp_gt_files.sort()
            
            if len(self.raw_img_files) != len(self.sp_gt_files):
                raise ValueError('The number of images and sp gt files do not match')
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        raw_img = cv2.imread(self.raw_img_files[idx], cv2.IMREAD_UNCHANGED)
        if self.if_seg:
            seg_gt = cv2.imread(self.seg_gt_files[idx], cv2.IMREAD_UNCHANGED)
            sample = {'raw_img': raw_img, 'seg_gt': seg_gt}
        if self.if_sp:
            sp_gt = cv2.imread(self.sp_gt_files[idx], cv2.IMREAD_UNCHANGED)
            sample = {'raw_img': raw_img, 'sp_gt': sp_gt}
        if self.if_seg and self.if_sp:
            seg_gt = cv2.imread(self.seg_gt_files[idx], cv2.IMREAD_UNCHANGED)
            sp_gt = cv2.imread(self.sp_gt_files[idx], cv2.IMREAD_UNCHANGED)
            sample = {'raw_img': raw_img, 'seg_gt': seg_gt, 'sp_gt': sp_gt}
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/cityscapes/train/' 
    mydataset = FeatureFusionDataset(testPath, if_sp=False, if_seg=True)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        raw_img = data['raw_img'].numpy()
        seg_gt = data['seg_gt']
        pass