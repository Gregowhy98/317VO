import os
import pickle
import torch
import json
import time
import cv2
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from datasets.kittidataset import KittiDatasetMono

from models.xfeat.xfeat import XFeat

class TwoFrameVO:
    def __init__(self, configs, device):
        self.configs = configs
        self.device = device
        self.xfeat = XFeat(self.configs['xfeat']['model_path'])
        self.K = self._load_intrinsic(self.configs['kitti']['dataset_root_dir'], seq=self.configs['kitti']['sequence'])
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        
    @staticmethod
    def _load_intrinsic(folder, seq):
        calib_path = os.path.join(folder, seq, 'calib.txt')
        with open(calib_path, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K
    
    def get_abs_scale(self, gt_pre, gt_cur):     # gt_pre, gt_cur: 4x4 numpy array
        return np.sqrt(np.sum((gt_cur[0:3, 3] - gt_pre[0:3, 3])**2))
    
    def get_pose(self, img1, img2, pose_gt_1, pose_gt_2):
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)
        # pred1 = self.xfeat.detectAndCompute(img1, top_k = self.configs['xfeat']['sparse_keypoints'])[0]
        mtckpt1, mtckpt2 = self.xfeat.match_xfeat(img1, img2, top_k=self.configs['xfeat']['sparse_keypoints'])
        E, _ = cv2.findEssentialMat(mtckpt1, mtckpt2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, mtckpt1, mtckpt2, self.K)
        
        # get scale
        # if (scale > 0.09):
        #     t = t.reshape(3, 1) + scale * R @ t.reshape(3, 1)
        #     R = R @ pose_gt_2[0:3, 0:3]
        # t = t.reshape(3, 1) + scale * R @ t.reshape(3, 1)
        # R = R @ pose_gt_2[0:3, 0:3]
        print('R:', R, 't:', t)
        return R, t

def write_poses_to_file(r, t, filename):
    poses = np.concatenate((r, t), axis=1)
    poses = poses.reshape(1, 12)
    with open(filename, 'w') as f:
        for i in range(poses.shape[0]):
            f.write(' '.join([str(x) for x in poses[i]]) + '\n')
            
def write_pose_to_txt(r, t, txt_file):
    pose = np.concatenate((r, t), axis=1)
    pose_del = pose.flatten()
    str = '' 
    pose_del_len = len(pose_del)
    for idx, i in enumerate(pose_del):
        str += '%.6f' % i
        if idx < (pose_del_len - 1):
            str += ' '
    str += "\n"
    print(str)
    txt_file.writelines(str)
            
def evo(configs):
    if configs['is_evo']:
        os.system('evo_traj kitti result.txt --ref=./dataset/gt_pose/'+ configs['data_sequnce'] +'_gt.txt -p --plot_mode=xz')
        os.system('evo_ape kitti ./dataset/gt_pose/'+ configs['data_sequnce'] +'_gt.txt result.txt -va')
        os.system('evo_rpe kitti ./dataset/gt_pose/'+ configs['data_sequnce'] +'_gt.txt result.txt -va')
        cv2.destroyAllWindows()
        print('evo analysis done!')

def demo_vo():
    #=================================init=======================================
    config_path = '/home/wenhuanyao/317VO/configs/vodemo_configs.json'
    with open(config_path, 'r') as f:
        configs = json.load(f)
    device = torch.device(configs['device']+':'+configs['device_idx'] if torch.cuda.is_available() else "cpu")
    os.makedirs(configs['output_save_dir'], exist_ok=True)
    file_name = configs['demodataset']+'_'+ configs['kitti']['sequence']+'_result_' + str(time.time()) +'.txt'
    pose_save_file = os.path.join(configs['output_save_dir'], file_name)
    txt_file = open(pose_save_file, 'w')
    #=================================model=========================================
    vo = TwoFrameVO(configs, device)
    #=================================dataset=======================================
    mydataset = KittiDatasetMono(configs['kitti']['dataset_root_dir'], 
                                 seq=configs['kitti']['sequence'], 
                                 cam=configs['kitti']['cam'])
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)
    #===============================start demo===================================
    time_total = 0
    for i, data in tqdm(enumerate(mydatasetloader)):
        t_start = time.time()
        img1, img2, pose_gt_1, pose_gt_2 = data 
        R_esti, t_esti = vo.get_pose(img1, img2, pose_gt_1, pose_gt_2)
        write_pose_to_txt(R_esti, t_esti, txt_file) 
        t_end = time.time()
        time_total += t_end - t_start
        print('time:', t_end - t_start, 'average time:', time_total/(i+1))
    print('demo done!')
    print('total time:', time_total,'average time:', time_total/(i+1))
    # ==============================end demo & analysis===================================
    try:
        txt_file.close()
    except:
        print('result files write errors!')
    # evo(configs)
    
    
if __name__ == '__main__':
    demo_vo()