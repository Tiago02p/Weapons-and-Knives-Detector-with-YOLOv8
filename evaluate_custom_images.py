from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_custom_images(image_dir, output_dir='evaluation_results/custom_images'):
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load the model
        print("Loading YOLO model...")
        model = YOLO('./runs/detect/ak47_detection5/weights/best.pt')
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
            image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")
            
        print(f"\nFound {len(image_files)} images to evaluate")
        
        # Initialize metrics
        total_detections = 0
        total_images = len(image_files)
        detection_results = []
        
        # Process each image
        for img_path in image_files:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            # Run detection
            results = model(img, conf=0.5)  # Using 0.5 confidence threshold
            
            # Get detections for this image
            detections = results[0].boxes.data.cpu().numpy()
            num_detections = len(detections)
            total_detections += num_detections
            
            # Save annotated image
            annotated_img = results[0].plot()
            output_img_path = output_path / f"detected_{img_path.name}"
            cv2.imwrite(str(output_img_path), annotated_img)
            
            # Store results
            detection_results.append({
                'Image': img_path.name,
                'Detections': num_detections,
                'Confidence': detections[:, 4].mean() if num_detections > 0 else 0
            })
            
            print(f"Processed {img_path.name}: {num_detections} detections")
        
        # Create summary DataFrame
        results_df = pd.DataFrame(detection_results)
        
        # Calculate statistics
        avg_detections = total_detections / total_images
        images_with_detections = len([r for r in detection_results if r['Detections'] > 0])
        detection_rate = images_with_detections / total_images
        
        # Save results to CSV
        results_df.to_csv(output_path / 'detection_results.csv', index=False)
        
        # Create summary plot
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Number of detections per image
        plt.subplot(1, 2, 1)
        sns.barplot(data=results_df, x='Image', y='Detections')
        plt.xticks(rotation=45, ha='right')
        plt.title('Detections per Image')
        
        # Plot 2: Average confidence per image
        plt.subplot(1, 2, 2)
        sns.barplot(data=results_df, x='Image', y='Confidence')
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Confidence per Image')
        
        plt.tight_layout()
        plt.savefig(output_path / 'detection_summary.png')
        plt.close()
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Total images processed: {total_images}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {avg_detections:.2f}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Detection rate: {detection_rate:.2%}")
        print(f"\nResults saved to: {output_path}")
        print("Files generated:")
        print("- detection_results.csv: Detailed results for each image")
        print("- detection_summary.png: Summary plots")
        print("- detected_*.jpg: Annotated images with detections")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # You can change this to any directory containing your test images
    test_images_dir = "imgs/input" 
    evaluate_custom_images(test_images_dir) 