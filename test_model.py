import cv2
import numpy as np
import tensorflow as tf
import time

def test_tflite_model():
    print("Loading TFLite model...")
    # Load TFLite model and allocate tensors
    try:
        interpreter = tf.lite.Interpreter(model_path='yolov8n_float32.tflite')
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel Input Shape: {input_details[0]['shape']}")
    print(f"Model Output Shape: {output_details[0]['shape']}")

    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return

    # COCO classes
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
               'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    print("\nStarting detection... Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame!")
            break

        # Get start time for FPS calculation
        start_time = time.time()

        # Preprocess image
        input_shape = (640, 640)  # YOLOv8n input size
        img = cv2.resize(frame, input_shape)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process detections
        for detection in output_data[0]:
            confidence = float(detection[4])
            if confidence > 0.25:  # Confidence threshold
                class_id = int(detection[5])
                if class_id < len(CLASSES):
                    # Get coordinates (normalized to [0,1]) and convert to pixel values
                    x1, y1, x2, y2 = detection[0:4]
                    x1 = int(x1 * frame.shape[1])
                    y1 = int(y1 * frame.shape[0])
                    x2 = int(x2 * frame.shape[1])
                    y2 = int(y2 * frame.shape[0])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label
                    label = f'{CLASSES[class_id]}: {confidence:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('YOLOv8n TFLite Detection', frame)

        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopping detection...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("YOLOv8n TFLite Model Test")
    print("-------------------------")
    test_tflite_model()