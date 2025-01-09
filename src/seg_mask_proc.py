import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets.abandon.mydataset import FeatureFusionDataset
from collections import namedtuple


Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,        0 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       50 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,      126 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,      126 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      225 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,      230 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,      190 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,      180 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,      255 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,      255 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,      250 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,       90 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       40 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        0 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        0 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        0 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       50 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       50 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       50 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      220 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        0 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        0 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        0 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,        0 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def check_color_format(img):
    pixel = img[0, 0]
    if pixel[0] > pixel[2]:
        return "BGR"
    else:
        return "RGB"

def save_rewrite():
    pass

if __name__ == '__main__':
    dataset_folder = '/home/wenhuanyao/Dataset/cityscapes/'
    use = 'test'
    mydataset = FeatureFusionDataset(dataset_folder, use=use, if_sp=False, raw_seg=True, weighted_seg=False)
    mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)
    output_folder = os.path.join(dataset_folder, use, 'weighted_seg')
    
    class_dic = {}
    for obj in labels:
        trainid = obj.trainId
        color = obj.color
        class_dic[color] = trainid
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将 class_dic 转换为张量
    class_dic_tensor = torch.zeros((256, 256, 256), dtype=torch.uint8, device=device)
    for color, trainid in class_dic.items():
        class_dic_tensor[color[0], color[1], color[2]] = trainid
    
    for i, data in tqdm(enumerate(mydataset)):
        img = data['raw_seg']
        raw_img = data['raw_img']
        img_tensor = torch.tensor(img, device=device)
        
        # 将颜色转换为索引 RGB
        color_indices = img_tensor[:, :, :3].long()
        color_indices = color_indices.permute(2, 0, 1).contiguous()
        
        # 使用 class_dic_tensor 查找对应的 trainId BGR
        seg_gt = class_dic_tensor[color_indices[2], color_indices[1], color_indices[0]]
        
        # 将 seg_gt 转换为灰度图
        gray_seg_gt = seg_gt.cpu().numpy()
        
        #========================
        name = os.path.basename(data['raw_seg_path']).split('_')[:-2]
        new_name = '_'.join(name) + '_gtFine_weighted.png'
        write_path = os.path.join(output_folder, new_name)
        cv2.imwrite(write_path, gray_seg_gt)
        pass
    print('All Done!')
    
    
    
# =====================================cpu 处理========================================

# if __name__ == '__main__':
#     dataset_folder = '/home/wenhuanyao/Dataset/cityscapes/'
#     use = 'train'
#     mydataset = FeatureFusionDataset(dataset_folder, use=use, if_sp=False, raw_seg=True, weighted_seg=False)
#     mydatasetloader = DataLoader(mydataset, batch_size=1, shuffle=False)
#     output_folder = os.path.join(dataset_folder, use)
    
#     class_dic = {}
#     for obj in labels:
#         trainid = obj.trainId
#         color = obj.color
#         class_dic[color] = trainid
    
#     for i, data in tqdm(enumerate(mydataset)):
#         img = data['raw_seg']
#         raw_img = data['raw_img']
#         H, W, _ = img.shape
#         seg_gt = img.copy()
        
#         for h in range(H):
#             for w in range(W):
#                 color_ = tuple(int(c) for c in img[h, w][:3])
#                 color = (color_[2], color_[1], color_[0])
#                 if color in class_dic:
#                     seg_gt[h, w][3] = class_dic[color]
#                 else:
#                     seg_gt[h, w][3] = 0
                    
#         grey_seg_gt = seg_gt[:, :, 3]
#         # cv2.imwrite(os.path.join(output_folder, 'seg_gt', data['seg_gt_path']), seg_gt)
#         cv2.imwrite('test_seg_gt.png', grey_seg_gt)
#         # cv2.imwrite('raw_seg_gt.png', img)
#         # cv2.imwrite('raw_img.png', raw_img)
#         pass
#     print('All Done!')