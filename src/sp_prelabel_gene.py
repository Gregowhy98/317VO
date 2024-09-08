import os
import numpy
import cv2
import torch
from models.superpoint import SuperPointNet
import torch.nn as nn

# this file is used to generate the superpoint prelabel for training 

def create_directory(path, directory_name):
    full_path = os.path.join(path, directory_name)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        print(f"Directory '{directory_name}' created successfully")
    else:
        print(f"Directory '{directory_name}' already exists")

def predict_sp(img_path, sp):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 480))
    img = numpy.expand_dims(img, axis=0)
    img = numpy.expand_dims(img, axis=3)
    img = img.astype(numpy.float32) / 255.
    torch_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.device('cuda'))
    pred = sp.predict(img)
    return pred

class SuperPoint(nn.Module):
    def __init__(self):
        super(SuperPoint, self).__init__()
        pass
    def forward(self, x):
        return x

def save_pred(pred, target_folder_path, f):
    pass

def main():
    source_folder = '/home/wenhuanyao/Dataset/cityscapes_coco/val'
    target_path = '/home/wenhuanyao/Dataset/cityscapes_coco/sp_prelabel'
    target_folder_name = 'val'   # val/train
    create_directory(target_path, target_folder_name)
    
    target_folder_path = os.path.join(target_path, target_folder_name)
    
    files = os.listdir(source_folder)
    files.sort()
    # img_size = (1024, 2048)
    
    sp = SuperPointNet()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sp.to(device)
    sp.eval()
    sp.load_state_dict(torch.load('/home/wenhuanyao/317VO/pretrained/superpoint_v1.pth'))
    
    for f in files:
        if f.endswith('.png'):
            pred = predict_sp(os.path.join(source_folder, f), sp)
            save_pred(pred, target_folder_path, f)
            pass
    print('Done')
    pass

if __name__ == '__main__':
    main()
    pass