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

class CityScapesDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None):
        if use not in ['train', 'val', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, val, test]')
        
        self.dataset_folder = os.path.join(dataset_folder, use)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((800, 608)),
                transforms.ToTensor()
                ])
        
        # load raw image files
        raw_img_folder = os.path.join(self.dataset_folder, 'raw_img')
        self.raw_img_files = list_files(raw_img_folder)
        self.raw_img_files.sort()
        self.N = len(self.raw_img_files)
        print('find', self.N, 'raw imgs.')
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        # raw img 3 channels, 0-255, HxWxC, rgb
        raw_img = cv2.imread(self.raw_img_files[idx], cv2.COLOR_BGR2RGB)
        img_color = self.transform(raw_img)
        # img_color = torch.from_numpy(raw_img).permute(2, 0, 1).float()
        return {'img_color': img_color, 'path': self.raw_img_files[idx]} 
    
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/cityscapes/' 
    mydataset = CityScapesDataset(testPath, use='train', transform=None)
    mydatasetloader = DataLoader(mydataset, batch_size=4, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        raw_img = data['img_color']
        pass