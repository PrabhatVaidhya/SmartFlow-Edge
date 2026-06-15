import cv2
import numpy as np
import os
import time
import sys

# Ensure project root is in system path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vision.camera import CameraManager

def generate_sample():
    print("Initializing SmartFlow procedural video compiler...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    video_path = os.path.join(data_dir, "sample_fail.mp4")
    
    # Initialize camera manager in simulated mode
    cam = CameraManager(simulated=True)
    
    # Video writer setup (use mp4v codec which is standard on Windows/macOS/Linux)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    if not out.isOpened():
        print("Error: Could not open VideoWriter. Check OpenCV codec configurations.")
        return
        
    print("Generating 20 seconds of high-fidelity print defect footage (600 frames)...")
    
    # Frame generation loop
    for i in range(600):
        # Trigger defect after 150 frames (5 seconds) of clean print
        if i == 150:
            print(">> Critical anomaly event injected! Simulating spaghetti defect...")
            cam.trigger_defect(True)
            
        # Draw simulated frame
        frame = cam._generate_simulated_frame()
        out.write(frame)
        
        if i % 100 == 0:
            print(f"Progress: Compiled {i}/600 frames...")
            
    out.release()
    print(f"Success! Procedural print defect footage compiled successfully to: {video_path}")

if __name__ == "__main__":
    generate_sample()
