import os
import shutil
from pathlib import Path

def create_mini_dataset(src_dir, dest_dir, num_images):
    src_images_dir = src_dir / 'images'
    src_labels_dir = src_dir / 'labels'
    
    dest_images_dir = dest_dir / 'images'
    dest_labels_dir = dest_dir / 'labels'
    
    for split in ['train', 'val']:
        # If val, use a proportional amount (e.g., 20% of train size)
        # Assuming the request "200 images instead of 10" refers to the train split which had 10.
        split_num = num_images if split == 'train' else num_images // 2  # So 200 for train, 100 for val
        
        split_src_img = src_images_dir / split
        split_src_lbl = src_labels_dir / split
        
        split_dest_img = dest_images_dir / split
        split_dest_lbl = dest_labels_dir / split
        
        # Clear existing
        if split_dest_img.exists():
            shutil.rmtree(split_dest_img)
        if split_dest_lbl.exists():
            shutil.rmtree(split_dest_lbl)
            
        split_dest_img.mkdir(parents=True, exist_ok=True)
        split_dest_lbl.mkdir(parents=True, exist_ok=True)
        
        if not split_src_img.exists():
            print(f"Warning: {split_src_img} does not exist.")
            continue
            
        all_imgs = sorted(os.listdir(split_src_img))
        selected_imgs = all_imgs[:split_num]
        
        for img_name in selected_imgs:
            # Copy image
            shutil.copy2(split_src_img / img_name, split_dest_img / img_name)
            
            # Copy label if exists
            label_name = Path(img_name).stem + '.txt'
            src_lbl_path = split_src_lbl / label_name
            if src_lbl_path.exists():
                shutil.copy2(src_lbl_path, split_dest_lbl / label_name)
            else:
                # Create empty label file if it doesn't exist
                (split_dest_lbl / label_name).touch()
                
        print(f"Copied {len(selected_imgs)} images to {split_dest_img}")

if __name__ == '__main__':
    src = Path('/Users/techolon/Documents/csfiles/vip-team-2-drone-detection/data/seadronessee')
    dest = Path('/Users/techolon/Documents/csfiles/vip-team-2-drone-detection/data/seadronessee_mini')
    create_mini_dataset(src, dest, 200)
