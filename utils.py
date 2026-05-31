from PIL import Image
import os
import cv2

# --- 1. DATA CLEANING SECURITY GUARD ---
def check_and_clean_image(image_path, label_path):
    # Check A: Is the image corrupted or unreadable?
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Check B: Is the image way too small?
            if width < 100 or height < 100:
                print(f"⚠️ Skipping {image_path}: Image is too small ({width}x{height})")
                return False
    except Exception:
        print(f"❌ Skipping {image_path}: File is corrupted and cannot open")
        return False

    # Check C: Does it have a valid, non-empty label text file?
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        print(f"⚠️ Skipping {image_path}: Missing or empty label file")
        return False

    return True  # High quality photo!


# --- 2. CUSTOM DATA AUGMENTATION MAGIC (Pure OpenCV) ---
def augment_and_resize(img_path, lbl_path, save_img_path, save_lbl_path):
    """
    Resizes image to 512x512, flips it horizontally, 
    and recalculates the YOLO bounding boxes perfectly.
    """
    # Load original image
    image = cv2.imread(img_path)
    if image is None:
        return False
        
    h_orig, w_orig = image.shape[:2]
    
    # Resize image to 512x512 as requested by project details
    resized_image = cv2.resize(image, (512, 512))
    
    # Apply a Horizontal Flip (Left to Right Mirror)
    flipped_image = cv2.flip(resized_image, 1)
    
    # Read original YOLO coordinates and flip them
    flipped_bboxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w_box = float(parts[3])
                    h_box = float(parts[4])
                    
                    # 🔥 The Math Magic: Flipping an image horizontally means 
                    # the new X center becomes (1.0 - old X center)
                    new_x_center = 1.0 - x_center
                    
                    flipped_bboxes.append(f"{class_id} {new_x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")
    
    # Save the new flipped 512x512 image
    cv2.imwrite(save_img_path, flipped_image)
    
    # Save the new adjusted text bounding box file
    with open(save_lbl_path, 'w') as f:
        f.writelines(flipped_bboxes)
        
    return True