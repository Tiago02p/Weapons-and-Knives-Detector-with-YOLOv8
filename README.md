# Weapons-and-Knives Detector with YOLOv8

## Objectives
- Implement object detection using YOLOv8
- Create a real-time detection system
- Fine-tune a model for a custom dataset
- Evaluate detection performance (precision, recall, mAP)

## Tasks & Progress
1. **Set up YOLOv8 for real-time object detection from webcam**  
   - Implemented in `real_time_detection.py` (see below for usage)
2. **Create a custom dataset with annotated images (10-20 images per class)**  
   - Dataset structure established, using Roboflow export for AK-47 class
   - Images and labels organized in YOLO format under `datasets/dataset/`
3. **Fine-tune a pre-trained YOLO model on your custom dataset**  
   - Training script: `train_model.py` (see below for usage)
   - Model: `yolov8n.pt` (pretrained, fine-tuned on custom data)
4. **Implement tracking for detected objects across video frames**  
   - Tracking logic included in `real_time_detection.py`
5. **Evaluate the model using precision, recall, and mAP metrics**  
   - Metrics calculation and display in real-time detection script

---

## Installation

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - Key dependencies: `ultralytics`, `opencv-contrib-python`, `numpy`, `matplotlib`, `pyyaml`
3. **Download or prepare your dataset**
   - Place your dataset in `datasets/dataset/` following this structure:
     ```
     datasets/
       dataset/
         images/
           train/
           val/
         labels/
           train/
           val/
         data.yaml
     ```
   - Example `data.yaml`:
     ```yaml
     path: .
     train: images/train
     val: images/val
     names:
       0: AK-47
     ```

---

## Usage

### 1. Train the Model
```bash
python train_model.py
```
- This will fine-tune YOLOv8 on your custom dataset.
- Make sure your dataset is in the correct structure (see above).

### 2. Real-Time Detection from Webcam
```bash
python real_time_detection.py
```
- Detects AK-47 in real-time using your webcam.
- Displays bounding boxes, tracking, and live metrics (precision, recall, mAP).

---

## Description of Approach
- **Dataset**: Used Roboflow-exported dataset for AK-47 detection, organized in YOLO format. The system is extensible to more classes (e.g., knives) by adding images and updating `data.yaml`.
- **Training**: Fine-tuned a YOLOv8 nano model (`yolov8n.pt`) using the provided dataset. Training is handled by `train_model.py`, which also generates the correct `data.yaml`.
- **Detection & Tracking**: Real-time detection and object tracking are implemented in `real_time_detection.py` using OpenCV. The script draws bounding boxes, tracks objects across frames, and displays live performance metrics.
- **Evaluation**: Precision, recall, and mAP are calculated and shown during real-time detection.

---

## Challenges Overcome
- **Dataset Path Issues**: YOLOv8 expects a specific directory structure. Ensured the dataset is placed in `datasets/dataset/` and `data.yaml` uses `path: .` for correct relative paths.
- **Validation Set**: Created a script to populate the validation set from training images when missing.
- **Module Import Errors**: Resolved issues with Python environment and package installation to ensure `ultralytics` is available.
- **Windows Path Handling**: Adjusted scripts and documentation to work with Windows file paths and PowerShell.
- **Training Script Robustness**: Updated `train_model.py` to use absolute or correct relative paths, and to skip unnecessary dataset preparation steps if the dataset is already structured.

---

## Next Steps
- Expand dataset to include more weapon classes (e.g., knives)
- Further tune hyperparameters for improved accuracy
- Integrate additional evaluation and visualization tools

---

For any issues or questions, please refer to the code comments or open an issue in the repository.
