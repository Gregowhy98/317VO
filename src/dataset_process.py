import os
import numpy
from utils import draw_2d_mask
import cv2

color_map = {
    0: [0, 0, 0],  # Unlabeled area
    1: [70, 70, 70],  
    2: [190, 153, 153],  
    3: [250, 170, 160],  
    4: [220, 20, 60], 
}

weight_map = {
    0: 0,    # unlabeled
    1: 0.1,   # dynamic
    2: 0.1,
    3: 0.5,
    4: 1,    # static
}

class_conf = {
    'static': {1,2,3,7,9,10,16,18,19,21,23,27},
    'dynamic': {0,4,12,13,14,20,22,24,25,26},
    'other': {5,6,8,11,15},
}

def class_map(class_id, class_conf):
    if class_id in class_conf['static']:
        return 4
    elif class_id in class_conf['dynamic']:
        return 1
    elif class_id in class_conf['other']:
        return 3
    else:
        return 0

def create_directory(path, directory_name):
    full_path = os.path.join(path, directory_name)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        print(f"Directory '{directory_name}' created successfully")
    else:
        print(f"Directory '{directory_name}' already exists")

def generate_2d_mask(txt_file_path, target_folder, mask_size=(1024, 2048)):
    img = numpy.zeros(mask_size, dtype=numpy.uint8)
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split(' ')
            class_id = int(items[0])
            polygon = items[1:]
            cls_mp = class_map(class_id=class_id, class_conf=class_conf)
            img = draw_polygon(img, polygon, color_map[cls_mp], weight_map[cls_mp], thickness=1)
    return img

def draw_polygon(img_source, polygon, color, weight, thickness=1):
    h, w = img_source.shape
    img = img_source.copy()
    x_list = []
    y_list = []
    for idx, ele in enumerate(polygon):
        if idx % 2 == 0:
            x = int(float(ele) * h)
            x_list.append(x)
        else:
            y = int(float(ele) * w)
            y_list.append(y)
        polygon_scaled = numpy.array([[x, y] for x, y in zip(x_list, y_list)], numpy.int32)
        polygon_scaled = polygon_scaled.reshape((-1, 1, 2))
    img = cv2.fillPoly(img, [polygon_scaled], color, lineType=cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    return img

def main():
    source_folder = '/home/wenhuanyao/Dataset/cityscapes_coco/val'
    target_path = '/home/wenhuanyao/Dataset/cityscapes_coco/generate'
    target_folder_name = 'val'   # val/train
    create_directory(target_path, target_folder_name)
    
    target_folder_path = os.path.join(target_path, target_folder_name)
    
    files = os.listdir(source_folder)
    files.sort()
    txt_list = []
    img_size = (1024, 2048)
    
    for f in files:
        if f.endswith('.txt'):
            print(f)
            generate_2d_mask(f, target_folder_path, mask_size=img_size)
    print('Done')
    pass


if __name__ == '__main__':
    main()
    # f = '/home/wenhuanyao/Dataset/cityscapes_coco/val/munster_000173_000019_leftImg8bit.txt'
    # target_path = '/home/wenhuanyao/Dataset/cityscapes_coco/generate'
    # img_size = (1024, 2048)   # height, width
    # generate_2d_mask(f, target_path, mask_size=img_size)
    pass
    