import cv2
import os
import random
import glob
from collections import defaultdict

class BatchAugmentor:
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _get_random_transform(self):
        """Returns a single transform 'recipe' to be applied to a whole batch."""
        # Options: (type, code)
        transforms = [
            ('rotate', cv2.ROTATE_90_CLOCKWISE),
            ('rotate', cv2.ROTATE_180),
            ('rotate', cv2.ROTATE_90_COUNTERCLOCKWISE),
            ('flip', 1),  # Horizontal
            ('flip', 0),  # Vertical
            ('flip', -1)  # Both
        ]
        return random.choice(transforms)

    def _apply_specific_transform(self, image, transform_recipe):
        """Applies the chosen recipe to a single image."""
        t_type, t_code = transform_recipe
        if t_type == 'rotate':
            return cv2.rotate(image, t_code)
        elif t_type == 'flip':
            return cv2.flip(image, t_code)
        return image

    def augment_by_factor(self, factor=2):
        """
        Multiplies dataset. For each 'extra' set, 
        one uniform transform is picked for all images.
        """
        paths = glob.glob(os.path.join(self.source_dir, "*.*"))
        
        # 1. Save all originals first
        for p in paths:
            img = cv2.imread(p)
            cv2.imwrite(os.path.join(self.output_dir, f"orig_{os.path.basename(p)}"), img)

        # 2. Create 'n' augmented versions of the entire set
        for i in range(factor - 1):
            recipe = self._get_random_transform()
            recipe_name = f"{recipe[0]}_{recipe[1]}"
            
            for p in paths:
                img = cv2.imread(p)
                aug_img = self._apply_specific_transform(img, recipe)
                cv2.imwrite(os.path.join(self.output_dir, f"batch_{i}_{recipe_name}_{os.path.basename(p)}"), aug_img)

    def balance_dataset(self):
        """
        Balances classes. For each class needing more images, 
        it picks ONE transform and applies it to the needed amount of files.
        """
        label_map = defaultdict(list)
        for p in glob.glob(os.path.join(self.source_dir, "*.*")):
            label = os.path.basename(p).split('_')[0] 
            label_map[label].append(p)

        max_count = max(len(files) for files in label_map.values())

        for label, files in label_map.items():
            # Save originals
            for f in files:
                cv2.imwrite(os.path.join(self.output_dir, os.path.basename(f)), cv2.imread(f))
            
            # Augment minority classes using ONE consistent transform per class
            current_count = len(files)
            if current_count < max_count:
                recipe = self._get_random_transform() # Picked once for this specific label
                
                while current_count < max_count:
                    # Pick a source image from the original list and apply the class-wide transform
                    source_path = random.choice(files)
                    img = cv2.imread(source_path)
                    aug_img = self._apply_specific_transform(img, recipe)
                    
                    new_name = f"{label}_bal_type_{recipe[0]}_{current_count}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, new_name), aug_img)
                    current_count += 1