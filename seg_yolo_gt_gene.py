import os
import cv2
import numpy as np
import torch
import json

from models.yolo.net_inferencer import YOLOv8
from torch.utils.data import Dataset, DataLoader
from datasets.wireframedataset import WireframePrepocessDataset

def save_yolo_seg_gt(img_path, save_folder, pred):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    img_name = os.path.basename(img_path)
    save_path = os.path.join(save_folder, img_name)
    
    img = cv2.imread(img_path)
    # TODO: draw polygons
    
    # for i in range(len(polygons)):
    #     polygon = polygons[i]
    #     class_str = class_str_array[i]
    #     if class_str == 'line':
    #         color = (0, 0, 255)
    #     elif class_str == 'point':
    #         color = (0, 255, 0)
    #     else:
    #         color = (255, 0, 0)
    #     for j in range(len(polygon)):
    #         cv2.line(img, (polygon[j-1][0], polygon[j-1][1]), (polygon[j][0], polygon[j][1]), color, 2)
    
    # polygons0 = pred.masks.xy
        # class_idx_array0 = np.array(pred.boxes.cls.cpu().numpy(), dtype=int)
        # class_str_array0 = np.array([pred.names[i] for i in class_idx_array0])
    
    cv2.imwrite(save_path, img)
    print('save seg to:', save_path)

def run_yolo_seg(config_path):
    # load configs
    with open(config_path, 'r') as f:
        configs = json.load(f)
    device = torch.device(configs['device']+':'+ configs['device_idx'] if torch.cuda.is_available() else "cpu")
    
    # load model
    seg_model = YOLOv8(configs['seg_model_path'], device=device)
    class_groups = configs['class_groups']
    
    # load data
    mydataset = WireframePrepocessDataset(configs['dataset_folder'], use=configs['use'], transform=None)    # use: train or test
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)
    
    for i, data in enumerate(mydatasetloader):
        img_tensor, img_path = data[0], data[1]
        img_numpy = cv2.cvtColor(img_tensor.numpy(), cv2.COLOR_BGR2RGB)
        pred = seg_model.predict(img_numpy)[0]
        save_yolo_seg_gt(img_path[0], configs['save_folder'], pred)  
        pass
    print('yolo seg done!')

if __name__ == '__main__':
    config_path = '/home/wenhuanyao/317VO/configs/prepocess_configs.json'
    run_yolo_seg(config_path)

