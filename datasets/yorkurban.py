from torch.utils.data import Dataset
import scipy.io as sio
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pickle

class YorkUrbanDataset(Dataset):
    def __init__(self, dataset_folder, use='train', transform=None):
        if use not in ['train', 'test']:
            raise ValueError('Invalid value for use. Must be one of [train, test]')
        
        self.dataset_folder = dataset_folder
        self.use = use
        self.transform = transform
        
        # item list
        camPara_path = os.path.join(self.dataset_folder, 'cameraParameters.mat')
        camPara = sio.loadmat(camPara_path)
        list_mat_path = os.path.join(self.dataset_folder, 'ECCV_TrainingAndTestImageNumbers.mat')
        list_mat = sio.loadmat(list_mat_path)
        manhattan_mat_path = os.path.join(self.dataset_folder, 'Manhattan_Image_DB_Names.mat')
        manhattan_mat = sio.loadmat(manhattan_mat_path)
        
        folder_list = os.listdir(self.dataset_folder)
        self.img_list = [x for x in folder_list if x.startswith('P')]
        self.img_list = sorted(self.img_list)
        self.N = len(self.img_list)
        
        print('Number of images in {} set: {}'.format(self.use, self.N))
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # folder
        folder_path = os.path.join(self.dataset_folder, self.img_list[idx])
        # load img
        img_path = os.path.join(folder_path, self.img_list[idx] + '.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # load line mat
        line_mat_path = os.path.join(folder_path, self.img_list[idx] + 'LinesAndVP.mat')
        line_mat = sio.loadmat(line_mat_path)
        return {'img': img, 'line_mat': line_mat}
       
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/YorkUrbanDB' 
    # tforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((320, 640)),
    #     transforms.ToTensor()
    #     ])
    mydataset = YorkUrbanDataset(testPath, use='train', transform=None)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        ret = data
        pass