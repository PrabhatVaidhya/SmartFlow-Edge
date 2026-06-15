import os
import cv2
import random
import math
import numpy as np
from vision import VisionGateway

def patched_get_simulated_frame(self):
    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    frame[:] = (20, 22, 28) # Metallic grey background
    
    # Draw build grid lines (dark enough to be ignored by Canny threshold 176)
    for x in range(0, self.width, 40):
        cv2.line(frame, (x, 0), (x, self.height), (30, 32, 40), 1)
    for y in range(0, self.height, 40):
        cv2.line(frame, (0, y), (self.width, y), (30, 32, 40), 1)
        
    self.time_elapsed += 0.033
    if self.sim_layer < self.sim_max_layers:
        self.sim_layer += self.print_speed
        self.sim_nozzle_y = self.sim_bed_y - int(self.sim_layer * 1.0)
        self.sim_nozzle_x = self.sim_pillar_x + int(math.sin(self.time_elapsed * 10) * (self.sim_pillar_width // 2))
        
        if not self.defect_active:
            if int(self.sim_layer) % 2 == 0:
                self.sim_layer_lines.append((self.sim_pillar_x, int(self.sim_nozzle_y), self.sim_pillar_width))
        else:
            self.defect_severity = min(1.0, self.defect_severity + 0.005)
            
    # 1. Draw printed layers
    for px, py, pw in self.sim_layer_lines:
        cv2.line(frame, (px - pw // 2, py), (px + pw // 2, py), (0, 180, 80), 2)
        
    # 2. Draw active defect
    if self.defect_active and len(self.sim_layer_lines) > 0:
        top_healthy_y = self.sim_layer_lines[-1][1]
        if len(self.spaghetti_points) < 80:
            cx, cy = self.sim_nozzle_x, self.sim_nozzle_y
            curve = [(cx, cy)]
            for _ in range(random.randint(5, 12)):
                cx += random.randint(-18, 18)
                cy += random.randint(-4, 22)
                cx = max(10, min(self.width - 10, cx))
                cy = max(top_healthy_y - 20, min(self.sim_bed_y, cy))
                curve.append((cx, cy))
            self.spaghetti_points.append(curve)
            
        for curve in self.spaghetti_points[:int(len(self.spaghetti_points) * self.defect_severity)]:
            for i in range(len(curve) - 1):
                color = (0, int(120 + 135 * (1 - self.defect_severity)), int(180 + 75 * self.defect_severity))
                cv2.line(frame, curve[i], curve[i+1], color, 1, cv2.LINE_AA)
                
    # 3. Draw bed (hidden in background color)
    cv2.line(frame, (50, self.sim_bed_y), (self.width - 50, self.sim_bed_y), (20, 22, 28), 3)
    
    # 4. Draw toolhead (hidden in background color)
    # Hiding both keeps the contours 100% clean for physical feature engineering
    
    return frame

# Apply the monkey patch
VisionGateway.get_simulated_frame = patched_get_simulated_frame

def generate_synthetic_dataset(num_samples=500):
    print("Initializing synthetic 3D printing dataset generator...")
    
    # Create target directories
    os.makedirs("data/dataset/nominal", exist_ok=True)
    os.makedirs("data/dataset/defect", exist_ok=True)
    
    # Initialize the Vision Gateway
    gateway = VisionGateway()
    
    # 1. Generate Nominal (Healthy) Images
    print(f"Generating {num_samples} healthy/nominal print samples...")
    for i in range(num_samples):
        gateway.reset_simulation()
        gateway.trigger_defect(False)
        
        # Advance simulation to a random height
        target_layers = random.randint(30, 200)
        for _ in range(target_layers):
            gateway.sim_layer += gateway.print_speed
            gateway.sim_nozzle_y = gateway.sim_bed_y - int(gateway.sim_layer * 1.0)
            gateway.sim_nozzle_x = gateway.sim_pillar_x + int(math.sin(gateway.time_elapsed * 10) * (gateway.sim_pillar_width // 2))
            if int(gateway.sim_layer) % 2 == 0:
                gateway.sim_layer_lines.append((gateway.sim_pillar_x, int(gateway.sim_nozzle_y), gateway.sim_pillar_width))
            gateway.time_elapsed += 0.033
        
        # Capture the frame (without toolhead or bed)
        gateway.sim_layer = gateway.sim_max_layers
        frame = gateway.get_simulated_frame()
        file_path = f"data/dataset/nominal/nominal_{i:03d}.png"
        cv2.imwrite(file_path, frame)
        
    # 2. Generate Defect (Spaghetti) Images
    print(f"Generating {num_samples} spaghetti defect print samples...")
    for i in range(num_samples):
        gateway.reset_simulation()
        
        # Advance simulation to a random healthy height first
        target_layers = random.randint(30, 120)
        for _ in range(target_layers):
            gateway.sim_layer += gateway.print_speed
            gateway.sim_nozzle_y = gateway.sim_bed_y - int(gateway.sim_layer * 1.0)
            gateway.sim_nozzle_x = gateway.sim_pillar_x + int(math.sin(gateway.time_elapsed * 10) * (gateway.sim_pillar_width // 2))
            if int(gateway.sim_layer) % 2 == 0:
                gateway.sim_layer_lines.append((gateway.sim_pillar_x, int(gateway.sim_nozzle_y), gateway.sim_pillar_width))
            gateway.time_elapsed += 0.033
                
        # Trigger spaghetti defect
        gateway.trigger_defect(True)
        # Advance to let the defect grow with random severity
        defect_growth_steps = random.randint(40, 100)
        gateway.defect_severity = 0.0
        for _ in range(defect_growth_steps):
            gateway.sim_layer += gateway.print_speed
            gateway.sim_nozzle_y = gateway.sim_bed_y - int(gateway.sim_layer * 1.0)
            gateway.sim_nozzle_x = gateway.sim_pillar_x + int(math.sin(gateway.time_elapsed * 10) * (gateway.sim_pillar_width // 2))
            gateway.defect_severity = min(1.0, gateway.defect_severity + random.uniform(0.01, 0.03))
            gateway.get_simulated_frame()
            
        # Capture the frame (without toolhead or bed)
        gateway.sim_layer = gateway.sim_max_layers
        frame = gateway.get_simulated_frame()
        file_path = f"data/dataset/defect/defect_{i:03d}.png"
        cv2.imwrite(file_path, frame)
        
    print(f"Successfully generated synthetic dataset: {num_samples} nominal images and {num_samples} defect images.")

def generate_dataset_metadata_csv(nominal_dir="data/dataset/nominal", defect_dir="data/dataset/defect", csv_path="data/dataset/dataset_metadata.csv"):
    import csv
    from src.vision.detector import DefectDetector
    detector = DefectDetector()
    
    print(f"Compiling Kaggle-ready descriptor metadata CSV at: {csv_path}...")
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_label", "class_name", "deviation_pct", "complexity_score", "contour_count"])
        
        # Nominal folder features
        for img_name in sorted(os.listdir(nominal_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(nominal_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    _, telemetry = detector.analyze_frame(img)
                    writer.writerow([
                        f"nominal/{img_name}",
                        0,
                        "nominal",
                        telemetry["deviation_pct"],
                        telemetry["complexity_score"],
                        telemetry["contour_count"]
                    ])
                    
        # Defect folder features
        for img_name in sorted(os.listdir(defect_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(defect_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    _, telemetry = detector.analyze_frame(img)
                    writer.writerow([
                        f"defect/{img_name}",
                        1,
                        "defect",
                        telemetry["deviation_pct"],
                        telemetry["complexity_score"],
                        telemetry["contour_count"]
                    ])
    print("Metadata CSV successfully compiled!")

def zip_dataset(dataset_dir="data/dataset", zip_path="data/smartflow_defect_dataset.zip"):
    import zipfile
    print(f"Compressing dataset bundle into: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, os.path.dirname(dataset_dir))
                zipf.write(file_path, rel_path)
    print("Dataset ZIP archive successfully created!")

if __name__ == "__main__":
    # Generate 1,000 total images (500 nominal + 500 defect)
    generate_synthetic_dataset(num_samples=500)
    # Extract structural descriptors and write metadata CSV
    generate_dataset_metadata_csv()
    # Compress the directories into smartflow_defect_dataset.zip
    zip_dataset()
    print("\nSmartFlow Open-Source Defect Dataset Curated & Zipped Successfully! ready for Kaggle Flex!")
