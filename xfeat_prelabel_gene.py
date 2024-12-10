import os
import torch
import tqdm
import cv2
from torch.utils.data import DataLoader, Dataset
import pickle
from torchvision import transforms

from models.xfeat.xfeat import XFeat
from datasets.cityscapes import CityScapesDataset
from datasets.wireframe import WireframeDataset

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
    use = 'train'   # 'train' 'val' 'test'
    source_folder = os.path.join('/home/wenhuanyao/Dataset/cityscapes/', use)
    target_path = os.path.join(source_folder, 'xfeat')
    os.makedirs(target_path, exist_ok=True)
    
    raw_img_folder = os.path.join(source_folder, 'raw_img')
    files = os.listdir(raw_img_folder)
    files.sort()
    print('find', len(files), 'raw imgs.')
    
    # init xfeat
    xfeat_weights = '/home/wenhuanyao/317VO/pretrained/xfeat.pt'
    xfeat = XFeat(weights=xfeat_weights)
    
    # data prep
    myDataset = CityScapesDataset('/home/wenhuanyao/Dataset/cityscapes/', use=use, transform=None)
    myDatasetLoader = DataLoader(myDataset, batch_size=1, shuffle=False)
    
    for i, data in tqdm.tqdm(enumerate(myDatasetLoader), desc='processing'):
        raw_img = data['img_color']
        name = os.path.basename(str(data['path'][0])).strip().replace('.png','_xfeat.pkl')
        x = xfeat.detectAndCompute(raw_img, top_k = 2000)
        pred = x[0]
        dump_path = os.path.join(target_path, name)
        dump_data(pred, dump_path)
        
    print('Done')

 
def wireframe_xfeat_gene():
    use = 'train'   # 'train' 'test'
    dataset_folder = '/home/wenhuanyao/Dataset/Wireframe'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    gt_folder_path = os.path.join(dataset_folder, 'xfeat_gt')
    source_folder = os.path.join(gt_folder_path, use)
    os.makedirs(source_folder, exist_ok=True)
    
    # init xfeat
    xfeat_weights = '/home/wenhuanyao/317VO/pretrained/xfeat.pt'
    xfeat = XFeat(weights=xfeat_weights).to(device)
    
    # data prep
    myDataset = WireframeDataset(dataset_folder, use=use, transform=None)
    myDatasetLoader = DataLoader(myDataset, batch_size=1, shuffle=False)
    
    for i, data in tqdm.tqdm(enumerate(myDatasetLoader), desc='processing'):
        img = data['img']
        x = xfeat.detectAndCompute(img, top_k = 2000)
        pred = x[0]
        dump_path = str(data['xfeat_path'][0]).strip()
        dump_data(pred, dump_path)
        
    print('Done')
    


if __name__ == '__main__':
    wireframe_xfeat_gene()
    pass

