
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
        if use not in ['train', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, test]')
        
        self.dataset_folder = dataset_folder
        self.use = use
        self.transform = transform
        
        # item list
        v11_folder = os.path.join(dataset_folder, 'v1.1')
        list_file_path = os.path.join(v11_folder, self.use + '.txt')
        with open(list_file_path, 'r') as f:
            self.img_list = f.readlines()
        self.img_list = [x.strip() for x in self.img_list]
        self.N = len(self.img_list)
        print('Number of images in {} set: {}'.format(self.use, self.N))
        
        # folders
        self.img_folder = os.path.join(v11_folder, self.use)
        self.line_mat_folder = os.path.join(self.dataset_folder, 'line_mat')
        self.point_line_pkl_folder = os.path.join(self.dataset_folder, 'pointlines')
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # load img
        img_path = os.path.join(self.img_folder, self.img_list[idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # load line mat
        line_mat_path = os.path.join(self.line_mat_folder, self.img_list[idx].replace('.jpg', '_line.mat'))
        line_mat = sio.loadmat(line_mat_path).get('lines')
        # load point line pkl
        point_line_pkl_path = os.path.join(self.point_line_pkl_folder, self.img_list[idx].replace('.jpg', '.pkl'))
        point_line_pkl_data = pickle.load(open(point_line_pkl_path, 'rb'))
        
        return {'img': img, 'line_mat': line_mat, 'point_line_pkl': point_line_pkl_data}
       
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/Wireframe/' 
    # tforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((320, 640)),
    #     transforms.ToTensor()
    #     ])
    # mydataset = WireframeDataset(testPath, use='train', transform=None)
    mydataset = WireframeDataset(testPath, use='test', transform=None)

    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        ret = data
        pass