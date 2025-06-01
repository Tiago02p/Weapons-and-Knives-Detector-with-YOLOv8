import os
import shutil

train_img_dir = 'dataset/images/train'
val_img_dir = 'dataset/images/val'
train_label_dir = 'dataset/labels/train'
val_label_dir = 'dataset/labels/val'

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

img_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_file in img_files[:10]:
    # Copy image
    shutil.copy2(os.path.join(train_img_dir, img_file), os.path.join(val_img_dir, img_file))
    # Copy label
    label_file = os.path.splitext(img_file)[0] + '.txt'
    src_label = os.path.join(train_label_dir, label_file)
    dst_label = os.path.join(val_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy2(src_label, dst_label)
    else:
        print(f'Warning: Label file not found for {img_file}')

print('Validation set populated with 10 images and labels.') 