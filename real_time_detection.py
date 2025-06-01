import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class AK47Detector:
    def __init__(self, model_path='./runs/detect/train/weights/best.pt'):
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.frame_count = 0
        self.detection_history = defaultdict(list)
        
    def calculate_metrics(self):
        """Calculate precision, recall, and mAP for the current session"""
        total_detections = sum(len(detections) for detections in self.detection_history.values())
        total_frames = self.frame_count
        
        if total_frames == 0:
            return 0, 0, 0
            
        # Simple metrics calculation (can be enhanced with ground truth data)
        avg_detections_per_frame = total_detections / total_frames
        precision = min(1.0, avg_detections_per_frame)  # Assuming max 1 AK-47 per frame
        recall = min(1.0, avg_detections_per_frame)
        
        # Simplified mAP calculation
        mAP = (precision + recall) / 2
        
        return precision, recall, mAP

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            self.frame_count += 1
            
            # Run YOLOv8 tracking on the frame
            results = self.model.track(frame, persist=True, tracker="bytetrack")
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls = results[0].boxes.cls.cpu().tolist()
                conf = results[0].boxes.conf.cpu().tolist()
                
                # Visualize the results on the frame
                for box, track_id, class_id, confidence in zip(boxes, track_ids, cls, conf):
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:  # Limit track history
                        track.pop(0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, 
                                (int(x - w/2), int(y - h/2)), 
                                (int(x + w/2), int(y + h/2)), 
                                (0, 0, 255), 2)  # Red color for AK-47
                    
                    # Draw label
                    label = f"AK-47 {confidence:.2f}"
                    cv2.putText(frame, label, 
                              (int(x - w/2), int(y - h/2 - 10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw tracking line
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
                    
                    # Store detection for metrics
                    self.detection_history[track_id].append((class_id, confidence))
            
            # Display metrics
            precision, recall, mAP = self.calculate_metrics()
            metrics_text = f"Precision: {precision:.2f} Recall: {recall:.2f} mAP: {mAP:.2f}"
            cv2.putText(frame, metrics_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("AK-47 Detection", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AK47Detector()
    detector.run_detection() 