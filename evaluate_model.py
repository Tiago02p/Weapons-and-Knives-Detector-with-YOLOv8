from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os

def evaluate_model():
    try:
        # Load the model
        print("Loading YOLO model...")
        model = YOLO('./runs/detect/ak47_detection5/weights/best.pt')
        
        # Check if validation data exists
        val_images_path = Path('custom_dataset/images')
        if not val_images_path.exists():
            raise FileNotFoundError(f"Validation images not found at {val_images_path}")
            
        # Create a temporary data.yaml for validation
        temp_data_yaml = {
            'path': str(Path.cwd() / 'custom_dataset'),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': 1,
            'names': ['AK-47 Gun']
        }
        
        # Save temporary data.yaml
        temp_yaml_path = Path('temp_data.yaml')
        with open(temp_yaml_path, 'w') as f:
            for key, value in temp_data_yaml.items():
                if isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print("Running validation...")
        # Run validation on the validation dataset
        results = model.val(data=str(temp_yaml_path))
        
        # Get metrics as properties
        mp = results.box.mp  # Mean precision
        mr = results.box.mr  # Mean recall
        map50 = results.box.map50  # mAP at IoU 0.5
        map = results.box.map  # mAP at IoU 0.5:0.95
        
        # Print detailed metrics
        print("\n=== Model Evaluation Results ===")
        print(f"mAP50: {map50:.3f}")
        print(f"mAP50-95: {map:.3f}")
        print(f"Precision: {mp:.3f}")
        print(f"Recall: {mr:.3f}")
        
        # Create a directory for evaluation results if it doesn't exist
        eval_dir = Path('evaluation_results')
        eval_dir.mkdir(exist_ok=True)
        
        # Save metrics to a CSV file
        metrics_df = pd.DataFrame({
            'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall'],
            'Value': [map50, map, mp, mr]
        })
        metrics_df.to_csv(eval_dir / 'metrics.csv', index=False)
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(10, 8))
        confusion_matrix = results.confusion_matrix.matrix
        # Convert to percentage if the values are not already
        if confusion_matrix.max() > 1:
            confusion_matrix = confusion_matrix / confusion_matrix.sum() * 100
        
        sns.heatmap(confusion_matrix, 
                    annot=True, 
                    fmt='.1f',  # Format as float with 1 decimal place
                    cmap='Blues')
        plt.title('Confusion Matrix (%)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(eval_dir / 'confusion_matrix.png')
        plt.close()
        
        # Create and save metrics bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Metric', y='Value')
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.savefig(eval_dir / 'metrics_plot.png')
        plt.close()
        
        print("\nEvaluation results have been saved to the 'evaluation_results' directory")
        print("Files generated:")
        print("- metrics.csv: Detailed metrics in CSV format")
        print("- confusion_matrix.png: Visual representation of the confusion matrix")
        print("- metrics_plot.png: Bar plot of the main metrics")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_yaml_path.exists():
            temp_yaml_path.unlink()

if __name__ == "__main__":
    evaluate_model() 