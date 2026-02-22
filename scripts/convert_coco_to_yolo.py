"""
Convert SeaDronesSee COCO-format annotations to YOLO format.

COCO bbox: [x_min, y_min, width, height] in absolute pixels
YOLO bbox: [class_id, x_center, y_center, width, height] normalized 0-1
"""
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo(json_path, images_dir, output_images_dir, output_labels_dir):
    """Convert one split (train or val) from COCO to YOLO format."""

    # Load the COCO JSON file
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Build mapping: COCO category_id -> sequential YOLO class index
    # COCO IDs can be non-sequential (e.g., 1, 3, 5, 7, 10)
    # YOLO needs them as 0, 1, 2, 3, 4
    categories = sorted(coco['categories'], key=lambda c: c['id'])
    cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(categories)}

    print('Category mapping (COCO ID -> YOLO ID -> Name):')
    for cat in categories:
        print(f"  {cat['id']} -> {cat_id_to_yolo[cat['id']]} -> {cat['name']}")

    # Build mapping: image_id -> image info
    images = {img['id']: img for img in coco['images']}

    # Group annotations by image_id
    img_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    converted = 0
    skipped = 0

    for img_id, img_info in tqdm(images.items(), desc='Converting'):
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']

        # Copy image to output directory
        src = os.path.join(images_dir, file_name)
        dst = os.path.join(output_images_dir, file_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

        # Convert annotations for this image
        label_name = Path(file_name).stem + '.txt'
        label_path = os.path.join(output_labels_dir, label_name)
        annotations = img_annotations.get(img_id, [])

        lines = []
        for ann in annotations:
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_yolo:
                skipped += 1
                continue

            yolo_class = cat_id_to_yolo[cat_id]
            x_min, y_min, w, h = ann['bbox']

            # Convert: center_x, center_y, w, h (all normalized 0-1)
            x_center = (x_min + w / 2) / img_w
            y_center = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            lines.append(
                f'{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}'
            )

        # Write label file (empty file = no objects in this image)
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
        converted += 1

    print(f'Converted {converted} images, skipped {skipped} annotations')


if __name__ == '__main__':
    BASE = Path('data/raw')
    OUT = Path('data/seadronessee')

    print('=== Converting TRAINING set ===')
    convert_coco_to_yolo(
        json_path=BASE / 'instances_train.json',
        images_dir=BASE / 'images_train',
        output_images_dir=OUT / 'images' / 'train',
        output_labels_dir=OUT / 'labels' / 'train'
    )

    print('\n=== Converting VALIDATION set ===')
    convert_coco_to_yolo(
        json_path=BASE / 'instances_val.json',
        images_dir=BASE / 'images_val',
        output_images_dir=OUT / 'images' / 'val',
        output_labels_dir=OUT / 'labels' / 'val'
    )