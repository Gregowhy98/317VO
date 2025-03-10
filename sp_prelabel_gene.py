import os
import numpy as np
import cv2
import torch
from models.superpoint import SuperPointNet, pred_semi_desc, pred_sp_front, SuperPointFrontend
import torch.nn as nn

# this file is used to generate the superpoint prelabel for training 

def predict_sp(img_path, sp):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    inp = inp.cuda()
    pred = sp(inp)
    return pred

def pred_sp_front(img_path, sp_front):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    pts, desc, heatmap = sp_front.run(img)
    return pts, desc, heatmap

def save_pred(pred, target_folder_path, f):
    pass

def gene_sp_gt_cityscapes():
    use = 'train'
    source_folder = os.path.join('/home/wenhuanyao/Dataset/cityscapes', use)
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
    
    sp_front = SuperPointFrontend(weights_path, 4, 0.5, 0.7, True)
    
    for f in files:
        if f.endswith('.png'):
            f_path = os.path.join(img_folder, f)
            pred = predict_sp(f_path, sp)
            semi, desc = pred[0], pred[1]
            pts, desc_, heatmap = pred_sp_front(f_path, sp_front)
            # save_pred(pred, target_path, f)
            pass
    print('Done')
    
# def gene_sp_gt_wireframe():
#     use = 'train'
#     source_folder = '/home/wenhuanyao/Dataset/Wireframe'
#     img_folder = os.path.join(source_folder, 'v1.1/all')
#     target_path = os.path.join(source_folder, 'sp_gt')
#     os.makedirs(target_path, exist_ok=True)
    
#     files = os.listdir(img_folder)
#     files.sort()
    
#     weights_path = '/home/wenhuanyao/317VO/pretrained/superpoint_v1.pth'
#     sp = SuperPointNet()
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     sp.to(device)
#     sp.eval()
#     sp.load_state_dict(torch.load(weights_path))
    
#     # sp_front = SuperPointFrontend(weights_path, 4, 0.5, 0.7, True)
    
#     for f in files:
#         if f.endswith('.jpg'):
#             f_path = os.path.join(source_folder, f)
#             pred = predict_sp(f_path, sp)
#             desc = pred[0]
#             heatmap = pred[1]
#             # pts, desc, heatmap = pred_sp_front(f_path, sp_front)
#             # save_pred(pred, target_path, f)
#             pass
#     print('Done')

if __name__ == '__main__':
    # gene_sp_gt_wireframe()
    gene_sp_gt_cityscapes()
    pass