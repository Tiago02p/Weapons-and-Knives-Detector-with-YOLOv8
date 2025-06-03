import cv2
from ultralytics import YOLO
import numpy as np
import time

def real_time_detection():
    try:
        # Initialize YOLO model
        print("Loading YOLO model...")
        model = YOLO('./runs/detect/ak47_detection5/weights/best.pt')
        
        # Initialize webcam
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize FPS calculation variables
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        print("Starting detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:  # Update FPS every 30 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.5)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Add FPS counter to frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the annotated frame
            cv2.imshow("Real-time Detection", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    real_time_detection() 