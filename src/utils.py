
import json
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

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

def compute_score(pred, gt):
    score = 0
    return score

# def eval_loop(val_loader, model, criterion, device):
#     model.eval()
#     val_loss = 0
#     val_score_t = 0
#     val_score_r = 0

#     with torch.no_grad():
#         for i, data in enumerate(tqdm(val_loader, desc="Evaluating", ncols=100)):
#             img_color = data['img_color'].to(device)
#             seg_gt = data['seg_gt'].to(device)
#             img_gray = data['img_gray'].to(device)

#             # Forward pass
#             outputs = model(img_color)
#             pred_semi = outputs[0]
#             pred_desc = outputs[1]

#             # Compute loss
#             semi_loss = criterion(pred_semi, )
#             desc_loss = criterion(pred_desc, img_gray)
#             loss = semi_loss + desc_loss

#             # Accumulate loss
#             val_loss += loss.item()

#             # Compute evaluation metrics (example)
#             val_score += compute_score(pred_semi, )
#             val_score_t += compute_translation_score(pred_semi, seg_gt)
#             val_score_r += compute_rotation_score(pred_semi, seg_gt)

#     # Calculate average loss and scores
#     avg_val_loss = val_loss / len(val_loader)
#     avg_val_score = val_score / len(val_loader)
#     avg_val_score_t = val_score_t / len(val_loader)
#     avg_val_score_r = val_score_r / len(val_loader)

#     return avg_val_loss, avg_val_score, avg_val_score_t, avg_val_score_r


if __name__ == '__main__':
    txt_path = '/home/wenhuanyao/Dataset/cityscapes_coco/val/munster_000173_000019_leftImg8bit.txt'
    pic_path = '/home/wenhuanyao/Dataset/cityscapes_coco/val/munster_000173_000019_leftImg8bit.png'
    img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    ret = read_txt(txt_path)
    seg_idx, polygon = process_seg_gt(ret, img_size=(1024, 2048))
    mask = draw_2d_mask(img, polygon, color=(255,255,255))
    pass