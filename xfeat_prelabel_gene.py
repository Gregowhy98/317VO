
import os
import torch
import tqdm
import cv2
from torch.utils.data import DataLoader, Dataset
import pickle
from torchvision import transforms
import json
import numpy as np

from models.xfeat.xfeat import XFeat
from datasets.abandon.cityscapes import CityScapesDataset
from datasets.wireframedataset import WireframePrepocessDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_xfeat():
	xfeat_weights = '/home/wenhuanyao/317VO/pretrained/xfeat.pt'
	xfeat = XFeat(weights=xfeat_weights)
 
	# Batched mode
	x = torch.randn(4,3,480,640)
	outputs = xfeat.detectAndCompute(x, top_k = 2000)
	print("detected features on each batch item:", [len(o['keypoints']) for o in outputs])

def dump_data(pred, path):
    with open(path, 'wb') as f:
        pickle.dump(pred, f, pickle.HIGHEST_PROTOCOL)

def cityscapes_xfeat_gene():
    use = 'val'   # 'train' 'val' 'test'
    source_folder = os.path.join('/home/wenhuanyao/Dataset/cityscapes/', use)
    target_path = os.path.join(source_folder, 'xfeat_raw')
    os.makedirs(target_path, exist_ok=True)
    
    raw_img_folder = os.path.join(source_folder, 'raw_img')
    files = os.listdir(raw_img_folder)
    files.sort()
    print('find', len(files), 'raw imgs.')
    
    # init xfeat
    xfeat_weights = '/home/wenhuanyao/317VO/pretrained/xfeat.pt'
    xfeat = XFeat(weights=xfeat_weights).to(device)
    
    # data prep
    myDataset = CityScapesDataset('/home/wenhuanyao/Dataset/cityscapes/', use=use, transform=None)
    myDatasetLoader = DataLoader(myDataset, batch_size=1, shuffle=False)
    
    for i, data in tqdm.tqdm(enumerate(myDatasetLoader), desc='processing'):
        raw_img = data['img_color'].to(device)
        name = os.path.basename(str(data['path'][0])).strip().replace('.png','_xfeat.pkl')
        
        x = xfeat.detectAndCompute(raw_img, top_k = 2000)
        xx = xfeat.detectAndCompute_train(raw_img, top_k = 2000)
        dump_path = os.path.join(target_path, name)
        dump_data(x, dump_path)
        
    print('Done')

 
def wireframe_xfeat_gene(dataset_folder, use='train', pts_num=2000, xfeat_weights=None):
    gt_folder_path = os.path.join(dataset_folder, 'xfeat_gt')
    os.makedirs(gt_folder_path, exist_ok=True)
    
    # init xfeat
    xfeat = XFeat(weights=xfeat_weights).to(device)
    
    # data prep
    myDataset = WireframePrepocessDataset(dataset_folder, use=use, transform=None)
    myDatasetLoader = DataLoader(myDataset, batch_size=1, shuffle=False)
    
    for i, data in tqdm.tqdm(enumerate(myDatasetLoader), desc='processing'):
        img = data
        x = xfeat.detectAndCompute(img, top_k = pts_num)
        pred = x[0]
        pass
        # dump_path = str(data['xfeat_path'][0]).strip()
        # dump_data(pred, dump_path)
        
    print('Done')
    
def save_xfeat_kpts(kpts, save_path):
    kpts_np = kpts.cpu().numpy()
    np.save(save_path, kpts_np)
    
    
def wireframe_xfeat_kpts_gene(configs):
    kpts_save_path = os.path.join(configs['dataset_folder'], 'xfeat_kpts')
    os.makedirs(kpts_save_path, exist_ok=True)
    
    # init xfeat
    xfeat = XFeat(weights=configs['xfeat_kpts_gene']['model_path']).to(device)
    
    # data prep
    myDataset = WireframePrepocessDataset(configs['dataset_folder'], use='test', transform=None)
    myDatasetLoader = DataLoader(myDataset, batch_size=1, shuffle=False)
    
    for i, data in tqdm.tqdm(enumerate(myDatasetLoader), desc='processing'):
        img = data[0]
        name = os.path.basename(str(data[1][0])).strip().replace('.jpg','_kpts.npy')
        x = xfeat.detectAndCompute(img, top_k = configs['xfeat_kpts_gene']['max_points'])
        pred = x[0]
        kpts = pred['keypoints']
        save_xfeat_kpts(kpts, os.path.join(kpts_save_path, name))
    


if __name__ == '__main__':
    configs = '/home/wenhuanyao/317VO/configs/prepocess_configs.json'
    with open(configs, 'r') as f:
        configs = json.load(f)
        
    # wireframe_xfeat_gene(configs["dataset_folder"], use='train', pts_num=configs['xfeat']['max_keypoints'], xfeat_weights=configs['xfeat']['model_path'])
    # wireframe_xfeat_kpts_gene(configs)
    cityscapes_xfeat_gene()
    print('training data process done')


