import os
import random
import shutil
import cv2
# Import our custom functions from utils.py
from utils import check_and_clean_image, augment_and_resize

# Setup the input folder and where the output should go
source_dir = './data'       
output_dir = './dataset'

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create the empty standard folders for YOLO
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, folder, 'labels'), exist_ok=True)

# Gather all your raw images ending with .jpeg
all_images = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpeg')]

# Shuffle the images randomly so the split is completely fair
random.seed(42)
random.shuffle(all_images)

# Split math
total_images = len(all_images)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

# Helper function to process and move files safely
def move_and_augment_files(image_list, destination_split, should_augment=False):
    for img_name in image_list:
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + '.txt'
        
        src_img_path = os.path.join(source_dir, img_name)
        src_lbl_path = os.path.join(source_dir, label_name)
        
        # 1. Run Data Cleaning First!
        if check_and_clean_image(src_img_path, src_lbl_path):
            
            # Paths for standard split copy
            dest_img_path = os.path.join(output_dir, destination_split, 'images', img_name)
            dest_lbl_path = os.path.join(output_dir, destination_split, 'labels', label_name)
            
            # Copy original file to destination split
            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_lbl_path, dest_lbl_path)
            
            # 2. If it is the Training set, apply data augmentation to double dataset size!
            if should_augment:
                aug_img_name = f"{base_name}_aug.jpeg"
                aug_lbl_name = f"{base_name}_aug.txt"
                
                aug_img_path = os.path.join(output_dir, destination_split, 'images', aug_img_name)
                aug_lbl_path = os.path.join(output_dir, destination_split, 'labels', aug_lbl_name)
                
                # Create the augmented 512x512 variant
                augment_and_resize(src_img_path, src_lbl_path, aug_img_path, aug_lbl_path)

# Run the pipeline for all 3 groups
print("🚀 Starting Data Auditing, Cleaning, Resizing, and Splitting...")
move_and_augment_files(train_images, 'train', should_augment=True) # Augment training set!
move_and_augment_files(val_images, 'val', should_augment=False)
move_and_augment_files(test_images, 'test', should_augment=False)

print("\n📊 Cleaned & Augmented Dataset Summary:")
print(f"Total Original Images Found: {total_images}")
print(f"Images in Training folder (Original + Augmented): {len(os.listdir(os.path.join(output_dir, 'train', 'images')))}")
print(f"Images in Validation folder: {len(os.listdir(os.path.join(output_dir, 'val', 'images')))}")
print(f"Images in Testing folder: {len(os.listdir(os.path.join(output_dir, 'test', 'images')))}")
print("\n✅ Success! Your entire data preparation step is complete using pure OpenCV!")