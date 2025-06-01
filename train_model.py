from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import yaml

def prepare_dataset(source_dir, target_dir):
    """
    Prepare dataset from Roboflow format
    Args:
        source_dir: Directory containing Roboflow dataset
        target_dir: Directory to save the prepared dataset
    """
    # Create directory structure
    os.makedirs(os.path.join(target_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'val'), exist_ok=True)
    
    # Copy training images and labels
    train_img_dir = os.path.join(source_dir, 'images', 'train')
    train_label_dir = os.path.join(source_dir, 'labels', 'train')
    
    for img_file in os.listdir(train_img_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            # Copy image
            shutil.copy2(
                os.path.join(train_img_dir, img_file),
                os.path.join(target_dir, 'images', 'train', img_file)
            )
            # Copy corresponding label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(train_label_dir, label_file)):
                shutil.copy2(
                    os.path.join(train_label_dir, label_file),
                    os.path.join(target_dir, 'labels', 'train', label_file)
                )
    
    # Copy validation images and labels if they exist
    val_img_dir = os.path.join(source_dir, 'images', 'val')
    val_label_dir = os.path.join(source_dir, 'labels', 'val')
    
    if os.path.exists(val_img_dir):
        for img_file in os.listdir(val_img_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                # Copy image
                shutil.copy2(
                    os.path.join(val_img_dir, img_file),
                    os.path.join(target_dir, 'images', 'val', img_file)
                )
                # Copy corresponding label
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if os.path.exists(os.path.join(val_label_dir, label_file)):
                    shutil.copy2(
                        os.path.join(val_label_dir, label_file),
                        os.path.join(target_dir, 'labels', 'val', label_file)
                    )

def train_model(data_yaml_path, epochs=100, batch_size=16, img_size=640):
    """
    Train YOLOv8 model on custom dataset
    Args:
        data_yaml_path: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
    """
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=50,  # Early stopping patience
        save=True,    # Save best model
        device='cpu',  # Use CPU
        verbose=True,
        deterministic=True
    )
    
    return results

def create_data_yaml(dataset_path):
    """
    Create data.yaml file for YOLOv8 training
    Args:
        dataset_path: Path to dataset directory
    """
    data = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'AK-47'}
    }
    with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    # Dataset paths
    source_dir = './datasets/dataset'
    target_dir = './datasets/dataset'
    
    # Skip prepare_dataset since the dataset is already in the correct structure
    # prepare_dataset(source_dir, target_dir)
    
    # Create data.yaml
    create_data_yaml(target_dir)
    
    # Train model
    results = train_model(
        data_yaml_path=os.path.join(target_dir, 'data.yaml'),
        epochs=100,
        batch_size=16,
        img_size=640
    ) 