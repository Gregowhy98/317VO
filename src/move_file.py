import os
import shutil

origin_path = '/home/wenhuanyao/Dataset/cityscapes/gtFine/test'
target_path = '/home/wenhuanyao/Dataset/cityscapes/test/seg'
folder_list = os.listdir(origin_path)

for folder in folder_list:
    folder_path = os.path.join(origin_path, folder)
    file_list = os.listdir(folder_path)
    for f in file_list:
        if f.endswith('color.png'):
            source_path = os.path.join(folder_path, f)
            destination_path = os.path.join(target_path, f)
            shutil.move(source_path, destination_path)
        else:
            print(f'File {f} is not moved')
    

# png_path = '/home/wenhuanyao/Dataset/cityscapes/train/raw'
# txt_path = '/home/wenhuanyao/Dataset/cityscapes/train/txt'


# file_list = os.listdir(path)

# for f in file_list:
#     if f.endswith('.png'):
#         source_path = os.path.join(path, f)
#         destination_path = os.path.join(png_path, f)
#         shutil.move(source_path, destination_path)
#     elif f.endswith('.txt'):
#         source_path = os.path.join(path, f)
#         destination_path = os.path.join(txt_path, f)
#         shutil.move(source_path, destination_path)
#     else:
#         print(f'File {f} is not moved')

print('Done')