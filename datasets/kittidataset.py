
import numpy as np
import cv2
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import listdir


class KittiDatasetMono(Dataset):
    def __init__(self, kittifolder, seq='00', cam='0'):
        self.kittifolder = kittifolder
        self.seq = seq
        self.img_folder = os.path.join(self.kittifolder, self.seq, 'image_'+cam)
        
        # find imgs
        self.imgs_list = os.listdir(self.img_folder)
        self.imgs_list.sort()
        self.imgs_list = [os.path.join(self.img_folder, i.strip()) for i in self.imgs_list]
        print('find out', len(self.imgs_list), 'imgs.')
        
        # find poses
        pose_file = os.path.join(self.kittifolder, self.seq, 'gt_pose.txt')
        self.gt_poses = np.loadtxt(pose_file)
        print('find out', len(self.gt_poses), 'poses.')
        
        # num check
        if len(self.imgs_list) != len(self.gt_poses):
            print('numbers of imgs and poses do not match!')
        else:
            print('numbers of imgs and poses matched!')
        self.N = len(self.imgs_list)

    def __len__(self):
        return self.N - 1

    def __getitem__(self, idx):
        img1 = cv2.imread(self.imgs_list[idx], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(self.imgs_list[idx+1], cv2.IMREAD_UNCHANGED)
        
        return img1, img2, self.gt_poses[idx], self.gt_poses[idx+1]
    
if __name__ == '__main__':
    testPath = '/home/wenhuanyao/Dataset/KITTI'
    mydataset = KittiDatasetMono(testPath, seq='00', cam='0')
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)

    for i, data in enumerate(mydatasetloader):
        ret = data
        pass