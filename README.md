# Cauliflower Detection 🥦

This repository contains a specialized **YOLOv8n-seg** model trained for high-precision **Cauliflower** detection and instance segmentation. The project is designed for agricultural robotics and precision farming applications.

I am open-sourcing this model and dataset to allow the community to build upon this work, improve accuracy, and expand the dataset to benefit the agricultural AI community.

---

## 📂 Directory Structure

The repository is organized to maintain a clean workspace:

- **`base_models/`**: Contains the pretrained YOLO weights used as the starting point for training.
    - `yolov8n-seg.pt`: The original Ultralytics pretrained Nano segmentation model.
    - `yolo26n.pt`: An alternative Nano model variant.
- **`images/`**: Contains the source images used for the core training dataset. These images are **hand-annotated** using `labelme` for maximum precision.
- **`labels/`**: YOLO-format segmentation labels corresponding to the training images.
- **`scripts/`**: All Python utilities for training, data processing, and inference.
    - `detect_cauliflower.py`: The main inference script for testing the model.
    - `train_yolo.py`: Script used to train the model.
    - `split_dataset.py`: Utility to divide data into train/val sets.
    - `convert_labelme_to_yolo.py`: Converter for `labelme` JSONs to YOLO TXT format.
- **`auto_images/`**: A collection of cauliflower images that are **yet to be annotated**. These are kept aside for future dataset expansion.
- **`test_images/`**: A folder for raw images to verify and test model performance.
- **`runs/`**: Contains the training history, logs, and fine-tuned weights (`best.pt`).
- **`dataset.yaml`**: The configuration file defining paths and classes for YOLO.

---

## 📈 Model Performance & Results

The model has been fine-tuned for 80 epochs using the **YOLOv8n-seg** architecture.

### Training Metrics (from `runs/segment/train/`)
- **Box mAP@50:** `86.67%`
- **Mask mAP@50:** `85.74%`
- **Precision (Mask):** `87.26%`
- **Recall (Mask):** `79.70%`

The model produces high-quality segmentation masks that wrap tightly around the cauliflower heads, providing accurate centroid data for robotic systems.

### ⚠️ Current Challenges
While the model performs well on clear cauliflower heads, it occasionally misidentifies large-leafed weeds as cauliflowers in complex backgrounds. 

**Open Source Goal:** By sharing this model, I hope others can contribute more images of weeds and complex field conditions to the `images/` folder to help the model learn the difference and achieve even higher accuracy!

---

## 🚀 Usage

### 1. Download the Dataset Used to Train this Model here
You can download the hand-annotated dataset here:
(https://www.kaggle.com/datasets/yokshithkuchipudi/cauliflower-crop-dataset)

### 2. Run Inference
Use the dedicated inference script to test the model on your own images.

```bash
# Running from the root directory
python scripts/detect_cauliflower.py --model runs/segment/train/weights/best.pt --source test_image.jpg
```

---

## 🤝 Contributing
Contributions are highly encouraged! 
1. **Annotate**: Pick an image from `dataset_images/`, annotate it with `labelme`, and move it to `images/`.
2. **Train**: Run `python scripts/train_yolo.py` to refine the model with the new data.
3. **Submit**: Share your improved weights or new data via a Pull Request!
