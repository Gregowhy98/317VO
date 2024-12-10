
from torch.utils.data import Dataset
import scipy.io as sio
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pickle

class WireframeDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None):
        
        # init
        self.dataset_folder = dataset_folder
        if use not in ['train', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, test]')
        else:
            self.use = use
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((480, 640)),
                transforms.ToTensor()
                ])
        
        # folder path
        self.img_folder = os.path.join(self.dataset_folder, 'v1.1')
        self.line_mat_folder = os.path.join(self.dataset_folder, 'line_mat')
        self.point_line_pkl_folder = os.path.join(self.dataset_folder, 'pointlines')
        self.xfeat_gt_folder = os.path.join(self.dataset_folder, 'xfeat_gt')
        
        # item list
        list_file_path = os.path.join(self.img_folder, self.use + '.txt')
        with open(list_file_path, 'r') as f:
            item_list = f.readlines()
        item_list = sorted([item.strip() for item in item_list])
        self.N = len(item_list)
        print('Number of images in {} set: {}'.format(self.use, self.N))
        
        # find attributions
        self.img_list = [os.path.join(self.img_folder, self.use, x) for x in item_list]
        self.xfeat_gt_list = [os.path.join(self.xfeat_gt_folder, self.use, x.replace('.jpg', '_xfeat.pkl')) for x in item_list]
        self.line_mat_list = [os.path.join(self.line_mat_folder, x.replace('.jpg', '_line.mat')) for x in item_list]
        self.pointline_list = [os.path.join(self.point_line_pkl_folder, x.replace('.jpg', '.pkl')) for x in item_list]
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # load img
        img_path = self.img_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = self.transform(img)
        # load line mat
        line_mat_path = self.line_mat_list[idx]
        line_mat = sio.loadmat(line_mat_path).get('lines')
        # TODO: load point line pkl
        
        # load xfeat gt
        xfeat_gt_path = self.xfeat_gt_list[idx]
        # xfeat_gt = pickle.load(open(xfeat_gt_path), 'rb')
        
        return {'img': img_tensor, 'line_mat': line_mat, 'xfeat_path': xfeat_gt_path}
       
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/Wireframe/' 
    mydataset = WireframeDataset(testPath, use='train', transform=None)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        ret = data
        pass