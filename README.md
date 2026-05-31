# Crop-weed-detection

# AI-Powered Smart Crop & Weed Management System

An end-to-end Machine Learning and Data Engineering pipeline designed for precision agriculture. This system utilizes a custom-trained **YOLOv8** Object Detection model to scan field images, automatically distinguish between commercial crops and invasive weeds, and calculate localized infestation densities to drive autonomous localized spraying decisions.

---

##  Live Deployment
🔗 **Interactive Web Application:** [Deploying Live on Streamlit](https://crop-weed-detection-yolo-67bfxmgrtxwqhg67qufwna.streamlit.app/)  
🔗 **Source Code Repository:** [GitHub Repository](https://github.com/Chahat-Chouhan/crop_weed_project)

---

##  Core Architectural Pillars

The project is structured as a complete MLOps pipeline divided into four distinct phases:

### 1. Data Engineering & Integrity (`1_organize.py`)
* Developed automated validation scripts to filter out corrupted or unreadable images.
* Implemented a stratified split execution to partition raw imagery into standard `train` (80%), `val` (10%), and `test` (10%) structures.
* Dynamic dataset metadata mapping via a centralized `data.yaml` configuration profile.

### 2. Spatial Computer Vision Augmentation (`utils.py`)
* Programmed custom data augmentation matrices utilizing **OpenCV**.
* Engineered mathematical bounding-box recalculation engines to handle horizontal/vertical spatial coordinate shifts, artificially expanding the training sample size and preventing model overfitting.

### 3. Deep Learning Network Training (Google Colab)
* Scaled model computation by migrating local operations to cloud-based **NVIDIA GPU** architectures.
* Trained a highly optimized **YOLOv8 (You Only Look Once)** neural network architecture over 25 epochs at a high-resolution canvas scale ($640 \times 640$).
* Exported production-ready localized neural weights (`best.pt`).

### 4. Edge Inference & Analytical UI Engine (`4_app.py`)
* Built a client-facing web application interface utilizing **Streamlit**.
* Integrated RAM-caching decorators (`@st.cache_resource`) to maintain model availability for near-zero-latency local inference.
* Embedded an **Agronomy Decision Engine** that processes matrix bounding outputs to calculate real-time weed density index percentages and output prescriptive spot-spraying instructions.

---

## Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Deep Learning Framework:** Ultralytics YOLOv8
* **Image Processing & Math:** OpenCV, Pillow, NumPy
* **Frontend Web Framework:** Streamlit
* **Infrastructure:** Google Colab (GPU Training EC2 instances), GitHub (Version Control)

* 

---

## 📂 Project Repository Directory Layout

```text
crop_weed_project/
├── dataset/                  # Split dataset directories
│   ├── train/                # Training partition (Images + YOLO Bounding Text labels)
│   ├── val/                  # Validation partition
│   └── test/                 # Test partition
├── metrics/                  # Statistical charts downloaded from Colab
│   ├── results.png           # Loss and Accuracy performance metrics
│   └── confusion_matrix.png  # True vs Predicted verification matrix
├── 1_organize.py             # Data Engineering & partitioning automation script
├── utils.py                  # OpenCV dataset augmentation utilities
├── 4_app.py                  # Streamlit production frontend application script
├── best.pt                   # Trained model production neural weight layers
├── data.yaml                 # Configuration target file mapping paths and classes
└── requirements.txt          # Explicit environmental dependency registry file
