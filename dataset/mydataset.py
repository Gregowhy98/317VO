from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os

# todo: add other gt types

def list_png_filenames(directory):  
    png_filenames = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.lower().endswith('.png'):  
                png_filenames.append(file)
    return png_filenames  

def list_txt_filenames(directory):
    txt_filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_filenames.append(file)
    return txt_filenames


class FeatureFusionTrainDataset(Dataset):
    
    def __init__(self, img_folder, transform=None):
        
        self.img_folder = img_folder
        self.transform = transform
        
        # Load image files
        self.img_files = list_png_filenames(self.img_folder)
        self.img_files.sort()
        self.N = len(self.img_files) -1
        
        # Load seg gt files
        self.txt_files = list_txt_filenames(self.img_folder)
        self.txt_files.sort()
        
        if len(self.img_files) != len(self.txt_files):
            raise ValueError('The number of images and txt files do not match')
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        # Load image
        img = cv2.imread(self.img_folder + self.img_files[idx], cv2.IMREAD_UNCHANGED)
        # Load seg gt
        with open(self.img_folder + self.txt_files[idx], 'r') as f:
            seg_gt = f.readlines()
        seg_gt = np.array(seg_gt).tolist()
        
        sample = {'id': self.img_files[idx] ,'img': img, 'seg_gt': seg_gt}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

class FeatureFusionTestDataset(Dataset):
    
    def __init__(self, img_folder, transform=None):
        
        self.img_folder = img_folder
        self.transform = transform
        
        # Load image files
        self.img_files = list_png_filenames(self.img_folder)
        self.img_files.sort()
        self.N = len(self.img_files)
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        # Load image
        img = cv2.imread(self.img_folder + self.img_files[idx], cv2.IMREAD_UNCHANGED)
        
        sample = {'id': self.img_files[idx] ,'img': img}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/cityscapes_coco/test/' 
    mydataset = FeatureFusionTrainDataset(testPath)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        img = data['img'].numpy()
        seg_gt = data['seg_gt']
        img_id = data['id']
        print(img_id)