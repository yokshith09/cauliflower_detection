import os
import random
import shutil

base_dir = "images/train"
train_img_dir = "images/train"
val_img_dir = "images/val"

train_lbl_dir = "labels/train"
val_lbl_dir = "labels/val"

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(base_dir) if f.endswith(".jpg")]

random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

# Move validation images
for img in val_images:
    json_name = img.replace(".jpg", ".json")

    shutil.move(os.path.join(base_dir, img), os.path.join(val_img_dir, img))
    shutil.move(os.path.join(base_dir, json_name), os.path.join(val_img_dir, json_name))

print("Dataset split completed.")
