import os
import cv2
import torch
import pickle
import json

from tqdm import tqdm
from models.xfeat.xfeat import XFeat
from models.xfeat.model import *
from models.superpoint import SuperPointNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# def xfeat_gt_gene(img_path, target_folder):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     save_path = os.path.join(target_folder, os.path.basename(img_path).replace('.jpg', '_xfeat.pkl'))
    
#     configs = '/home/wenhuanyao/317VO/configs/prepocess_configs.json'
#     with open(configs, 'r') as f:
#         configs = json.load(f)
    
#     xfeat_weights = '/home/wenhuanyao/317VO/pretrained/xfeat.pt'
#     xfeat = XFeat(weights=xfeat_weights)
#     pred = xfeat.detectAndCompute(img, top_k = configs['xfeat']['sparse_keypoints'])
#     pred_dense = xfeat.detectAndComputeDense(img, top_k = configs['xfeat']['dense_keypoints'], multiscale=False)
    
    
#     output = pred[0]
    
#     # save pred
#     with open(save_path, 'wb') as f:
#         pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
#     print('save to', save_path)

def sp_gt_gene():
    use = 'train'
    source_folder = '/home/wenhuanyao/Dataset/cityscapes/train'
    img_folder = os.path.join(source_folder, 'raw_img')
    target_path = os.path.join(source_folder, 'sp_gt')
    os.makedirs(target_path, exist_ok=True)
    
    files = os.listdir(img_folder)
    files.sort()
    
    weights_path = '/home/wenhuanyao/317VO/pretrained/superpoint_v1.pth'
    sp = SuperPointNet()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sp.to(device)
    sp.eval()
    sp.load_state_dict(torch.load(weights_path))
    
    # sp_front = SuperPointFrontend(weights_path, 4, 0.5, 0.7, True)
    
    for f in files:
        if f.endswith('.png'):
            img_path = os.path.join(source_folder, f)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            pred = predict_sp(f_path, sp)
            desc = pred[0]
            heatmap = pred[1]
            # pts, desc, heatmap = pred_sp_front(f_path, sp_front)
            # save_pred(pred, target_path, f)
            pass
    print('Done')
    pass






if __name__ == '__main__':
    # source_folder = '/home/wenhuanyao/Dataset/Wireframe'
    # img_folder = '/home/wenhuanyao/Dataset/Wireframe/v1.1/all'
    # task = 'xfeat_gt' # 'xfeat_gt' 'sp_gt' 'yolo_gt'
    # target_folder = os.path.join(source_folder, task)
    # os.makedirs(target_folder, exist_ok=True)
    
    # img_list = os.listdir(img_folder)
    # img_list.sort()
    # img_list = [os.path.join(img_folder, img) for img in img_list]
    # print('find', len(img_list), 'imgs.')
    
    
    
    # for img_path in tqdm(img_list):
    #     if task == 'xfeat_gt':
    #         xfeat_gt_gene(img_path, target_folder)
    #     elif task == 'sp_gt':
    #         pass
    #     elif task == 'yolo_gt':
    #         pass
    #     pass
    
    sp_gt_gene()
    print('Done')
    pass