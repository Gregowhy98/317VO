
import json
import os
import numpy as np
import cv2

def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)    
    
def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        ret = f.readlines()
    return ret

def process_seg_gt(seg, img_size=(1024, 2048)):
    seg = seg.strip().split(' ')
    class_idx = seg[0]
    polygon_seq = seg[1:]
    return {'classidx': class_idx, 'polygon': polygon_seq}
    
def draw_2d_mask(img, poly, color):
    # GRAYSCALE IMG
    img_zero = np.zeros_like(img)
    mask = cv2.fillPoly(img_zero, poly, color=(255,255,255))
    cv2.imshow(mask)
    cv2.waitKey(1)
    return mask



if __name__ == '__main__':
    txt_path = '/home/wenhuanyao/Dataset/cityscapes_coco/val/munster_000173_000019_leftImg8bit.txt'
    pic_path = '/home/wenhuanyao/Dataset/cityscapes_coco/val/munster_000173_000019_leftImg8bit.png'
    img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    ret = read_txt(txt_path)
    seg_idx, polygon = process_seg_gt(ret, img_size=(1024, 2048))
    mask = draw_2d_mask(img, polygon, color=(255,255,255))
    pass