from ultralytics import YOLO
import cv2
import time

def run_webcam_detection():
    # Load YOLOv8n directly - no TFLite conversion needed for testing
    model = YOLO('yolov8n.pt')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            break
            
        # Get start time for FPS
        start_time = time.time()
        
        # Run detection
        results = model(frame, conf=0.25)  # Lower confidence threshold
        
        # Draw detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f'{model.names[int(class_id)]}: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('YOLOv8 Detection', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Just three simple steps
    print("1. Installing required package...")
    
    print("\n2. Starting webcam detection...")
    run_webcam_detection()