from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def prepare_dataset():
    """
    Prepare the dataset structure and create dataset.yaml file
    """
    # Create dataset.yaml file
    dataset_yaml = """
path: ./custom_dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

nc: 1
names: ['AK-47 Gun']

roboflow:
  workspace: yolo-1utwo
  project: ak47-detection
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/yolo-1utwo/ak47-detection/dataset/2
    """
    
    with open('custom_dataset/dataset.yaml', 'w') as f:
        f.write(dataset_yaml.strip())
    
    print("Dataset structure prepared successfully!")

def train_model():
    """
    Train the YOLOv8 model on the custom dataset
    """
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(
        data='custom_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='ak47_detection',
        patience=20,
        save=True,
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )
    
    print("Training completed!")

def main():
    # Prepare dataset structure
    prepare_dataset()
    
    # Train the model
    train_model()

if __name__ == "__main__":
    main() 