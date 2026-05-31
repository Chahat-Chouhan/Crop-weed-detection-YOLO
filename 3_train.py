from ultralytics import YOLO

def main():
    # 1. Load a pre-trained model (YOLOv8 nano is small, fast, and great for laptops)
    model = YOLO('yolov8n.pt')

    # 2. Start the training process
    print("Starting YOLO Model Training on Crop & Weed Dataset...")
    results = model.train(
        data='data.yaml',    # Points to the map file we just created
        epochs=25,           # How many times the model cycles through the dataset
        imgsz=512,           # Automatically resizes images to your required 512x512 matrix!
        batch=16,            # Number of images processed at once (adjust lower if laptop runs out of memory)
        device='cpu'         # Trains on CPU (change to '0' or 'cuda' if you have an Nvidia GPU)
    )
    print(" Training complete! Check the 'runs/detect/train' folder for your results.")

if __name__ == '__main__':
    main()