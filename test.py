# ✅ Install the necessary packages
import subprocess
import sys

# Function to install required packages
def install_packages():
    packages = [
        "ultralytics", 
        "tensorflow", 
        "onnx", 
        "onnxruntime", 
        "protobuf"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

# Install packages
install_packages()

# ✅ Step 1: Load YOLOv8 model from Ultralytics
from ultralytics import YOLO

# Load the YOLOv8n model
model_path = "yolov8n.pt"  # Replace with your own model path
model = YOLO(model_path)

# ✅ Step 2: Export the YOLOv8 model to TFLite format
model.export(format="tflite")

# Print success message
print("✅ YOLOv8 model exported to TFLite successfully!")
