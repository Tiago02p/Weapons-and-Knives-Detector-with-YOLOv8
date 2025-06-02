from ultralytics import YOLO
import cv2
import os

def test_model(image_path):
    # Load the trained model
    model = YOLO('runs/detect/ak47_detection5/weights/best.pt')
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run inference
    results = model(image)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"AK-47: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Create output directory if it doesn't exist
    os.makedirs('imgs/output', exist_ok=True)
    
    # Save the result
    output_path = os.path.join('imgs/output', f'result_{os.path.basename(image_path)}')
    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")
    
    # Display the image
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Test with a single image
    test_image = "imgs/input/image.jpg"  # Replace with your test image path
    test_model(test_image)
    
    # You can test multiple images by uncommenting and modifying this:
    # test_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
    # for img in test_images:
    #     test_model(img)

if __name__ == "__main__":
    main() 