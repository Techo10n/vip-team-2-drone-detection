import os
import random
import shutil

image_dir = "images/all"
label_dir = "labels/all"

train_img_dir = "images/train"
val_img_dir = "images/val"
train_lbl_dir = "labels/train"
val_lbl_dir = "labels/val"

images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
print("Total images found:", len(images))
random.shuffle(images)

split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
val_images = images[split_index:]

for img in train_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(train_img_dir, img))
    shutil.move(os.path.join(label_dir, img.replace(".jpg", ".txt")),
                os.path.join(train_lbl_dir, img.replace(".jpg", ".txt")))

for img in val_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(val_img_dir, img))
    shutil.move(os.path.join(label_dir, img.replace(".jpg", ".txt")),
                os.path.join(val_lbl_dir, img.replace(".jpg", ".txt")))

print("Done splitting dataset.")