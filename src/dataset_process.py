import os
import numpy
from utils import draw_2d_mask

def generate_2d_mask(txt_file_path, target_folder):
    pass

def create_directory(path, directory_name):
    full_path = os.path.join(path, directory_name)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        print(f"Directory '{directory_name}' created successfully")
    else:
        print(f"Directory '{directory_name}' already exists")

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
            generate_2d_mask(f, target_folder_path)
    pass


if __name__ is '__main__':
    main()
    pass
    