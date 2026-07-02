import cv2
import numpy as np
import time
import math
import random
import os
import pickle
from config import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX

def _load_model_cached(model_path):
    try:
        import streamlit as st
        @st.cache_resource
        def _read_pickle(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return _read_pickle(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)

class VisionGateway:
    """
    Module 1: The Vision Gateway (vision.py)
    Handles thread-safe video ingestion (Webcam, Loop File, or Procedural Simulator)
    and applies Edge Canny + Contour Complexity algorithms to detect spaghetti defects.
    """
    def __init__(self, camera_index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        # Calibration defaults
        self.canny_lower = 100
        self.complexity_threshold = 50.0
        self.warning_threshold = 15.0
        self.critical_threshold = 35.0
        self.nominal_width = 40.0
        self.deviation_history = []
        
        # Webcam/Stream Region of Interest (ROI) boundaries
        self.roi_enabled = False
        self.roi_x_min = 50
        self.roi_x_max = 590
        self.roi_y_min = 50
        self.roi_y_max = 430
        
        # Unified HIL Anomaly and Noise simulation variables
        self.anomaly_mode = "Spaghetti Effect"
        self.sensor_noise = 0.0
        self.layer_shift_offset = 0
        
        # State variables for Procedural Simulator
        self.sim_layer = 0
        self.sim_max_layers = 300
        self.sim_bed_y = self.height - 80
        self.sim_pillar_x = self.width // 2
        self.sim_pillar_width = 40
        self.sim_nozzle_x = self.width // 2
        self.sim_nozzle_y = self.sim_bed_y
        self.defect_active = False
        self.defect_severity = 0.0
        self.spaghetti_points = []
        self.sim_layer_lines = []
        self.print_speed = 0.8
        self.time_elapsed = 0.0

        # Load trained machine learning classifier (RBF-SVM v4.0) via cached utility
        self.model_path = "data/defect_classifier.pkl"
        self.clf = None
        self.model_version = "Offline (Heuristics)"
        self.ml_prediction = "NOMINAL"
        self.ml_probability = 0.0
        
        try:
            if os.path.exists(self.model_path):
                payload = _load_model_cached(self.model_path)
                if isinstance(payload, dict) and "model" in payload:
                    self.clf = payload["model"]
                    self.model_version = payload.get("version", "4.0-realworld-combined")
                else:
                    self.clf = payload
                    self.model_version = "v1.0-legacy"
                print(f"Loaded trained RBF-SVM model: {self.model_version}")
        except Exception as e:
            print(f"[WARN] Failed to load SVM classifier: {e}")

    def trigger_defect(self, state: bool):
        """Triggers the spaghetti failure in the simulator."""
        self.defect_active = state
        if not state:
            self.defect_severity = 0.0
            self.spaghetti_points = []

    def reset_simulation(self):
        """Resets the 3D printing simulation state."""
        self.sim_layer = 0
        self.sim_nozzle_y = self.sim_bed_y
        self.sim_layer_lines = []
        self.spaghetti_points = []
        self.defect_active = False
        self.defect_severity = 0.0
        self.time_elapsed = 0.0

    def get_simulated_frame(self):
        """Generates a high-fidelity synthetic 3D printing feed with an emerging defect."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (20, 22, 28) # Metallic grey background
        
        # Draw build grid lines
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (35, 38, 48), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (35, 38, 48), 1)
            
        self.time_elapsed += 0.033
        if self.sim_layer < self.sim_max_layers:
            self.sim_layer += self.print_speed
            self.sim_nozzle_y = self.sim_bed_y - int(self.sim_layer * 1.0)
            
            # Apply lateral offset if layer shift defect is active
            current_shift = 0
            if self.defect_active:
                self.defect_severity = min(1.0, self.defect_severity + 0.005)
                if self.anomaly_mode == "Layer Shifting":
                    current_shift = int(self.defect_severity * 60)
                    self.layer_shift_offset = current_shift
            
            self.sim_nozzle_x = self.sim_pillar_x + current_shift + int(math.sin(self.time_elapsed * 10) * (self.sim_pillar_width // 2))
            
            if not self.defect_active:
                if int(self.sim_layer) % 2 == 0:
                    self.sim_layer_lines.append((self.sim_pillar_x, int(self.sim_nozzle_y), self.sim_pillar_width, 0, False))
            else:
                if self.anomaly_mode == "Layer Shifting":
                    if int(self.sim_layer) % 2 == 0:
                        self.sim_layer_lines.append((self.sim_pillar_x + current_shift, int(self.sim_nozzle_y), self.sim_pillar_width, current_shift, False))
                elif self.anomaly_mode == "Warping / Bed Adhesion Collapse":
                    if int(self.sim_layer) % 2 == 0:
                        self.sim_layer_lines.append((self.sim_pillar_x, int(self.sim_nozzle_y), self.sim_pillar_width, 0, False))
                elif self.anomaly_mode == "Under-Extrusion / Nozzle Clog":
                    if int(self.sim_layer) % 2 == 0:
                        self.sim_layer_lines.append((self.sim_pillar_x, int(self.sim_nozzle_y), self.sim_pillar_width, 0, True))
                else: # Spaghetti
                    if int(self.sim_layer) % 2 == 0:
                        self.sim_layer_lines.append((self.sim_pillar_x, int(self.sim_nozzle_y), self.sim_pillar_width, 0, False))
                
        # 1. Draw printed layers
        for line in self.sim_layer_lines:
            px, py, pw = line[0], line[1], line[2]
            offset_x = line[3] if len(line) > 3 else 0
            is_under = line[4] if len(line) > 4 else False
            
            # Draw curled base layers if warping is active
            if self.defect_active and self.anomaly_mode == "Warping / Bed Adhesion Collapse" and py >= (self.sim_bed_y - 40):
                layer_depth_factor = max(0.0, (40 - (self.sim_bed_y - py)) / 40.0)
                tilt = int(self.defect_severity * 15 * layer_depth_factor)
                cv2.line(frame, (px - pw // 2, py - tilt), (px, py), (0, 180, 80), 2)
                cv2.line(frame, (px, py), (px + pw // 2, py - tilt), (0, 180, 80), 2)
            elif is_under:
                # Under-extruded layers are drawn thin and dashed
                for seg_x in range(px - pw // 2 + offset_x, px + pw // 2 + offset_x, 8):
                    cv2.line(frame, (seg_x, py), (seg_x + 4, py), (0, 180, 80), 1)
            else:
                cv2.line(frame, (px - pw // 2 + offset_x, py), (px + pw // 2 + offset_x, py), (0, 180, 80), 2)
            
        # 2. Draw active Spaghetti defect curves
        if self.defect_active and self.anomaly_mode == "Spaghetti Effect" and len(self.sim_layer_lines) > 0:
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
                    
        # 3. Draw bed (offset by 20px to prevent morphological bridging with printed cylinder)
        cv2.line(frame, (50, self.sim_bed_y + 20), (self.width - 50, self.sim_bed_y + 20), (100, 105, 115), 3)
        
        # 4. Draw toolhead
        if self.sim_layer < self.sim_max_layers:
            ny, nx = self.sim_nozzle_y, self.sim_nozzle_x
            cv2.line(frame, (0, ny - 15), (self.width, ny - 15), (60, 65, 75), 2)
            cv2.rectangle(frame, (nx - 20, ny - 30), (nx + 20, ny - 10), (80, 85, 95), -1)
            pts = np.array([[nx - 6, ny - 10], [nx + 6, ny - 10], [nx, ny]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 120, 255) if not self.defect_active else (0, 0, 255))
            cv2.circle(frame, (nx, ny), 3, (180, 220, 255), -1)
            
        # 5. Inject Factory Sensor Noise (Gaussian Interference) if configured
        if self.sensor_noise > 0.0:
            noise = np.random.normal(0, self.sensor_noise, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        return frame

    def process_frame(self, frame, canny_lower=None, complexity_thresh=None, warning_thresh=None, critical_thresh=None):
        """
        Applies pre-processing, Canny edge detection, morphological bridging, 
        and bottom-vs-top width self-calibration to calculate the anomaly deviation score.
        """
        if canny_lower is not None: self.canny_lower = canny_lower
        if complexity_thresh is not None: self.complexity_threshold = complexity_thresh
        if warning_thresh is not None: self.warning_threshold = warning_thresh
        if critical_thresh is not None: self.critical_threshold = critical_thresh
        
        processed = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. Edge segmentation
        canny_upper = min(255, int(self.canny_lower * 1.5))
        edges = cv2.Canny(blurred, self.canny_lower, canny_upper)
        
        # Apply ROI cropping mask if enabled to filter background noise
        if self.roi_enabled:
            roi_mask = np.zeros_like(edges)
            cv2.rectangle(roi_mask, (self.roi_x_min, self.roi_y_min), (self.roi_x_max, self.roi_y_max), 255, -1)
            edges = cv2.bitwise_and(edges, roi_mask)
            
        # Crop out the moving simulator toolhead to prevent morphological merging
        if self.sim_layer > 0:
            toolhead_mask = np.ones_like(edges)
            cv2.rectangle(toolhead_mask, (0, 0), (self.width, self.sim_nozzle_y + 2), 0, -1)
            edges = cv2.bitwise_and(edges, toolhead_mask)
            
            # Cap the cylinder ends to form a closed contour during nominal printing
            if not self.defect_active:
                expected_w = self.sim_pillar_width
                expected_x = self.sim_pillar_x - expected_w // 2
                cv2.line(edges, (expected_x, self.sim_nozzle_y + 3), (expected_x + expected_w, self.sim_nozzle_y + 3), 255, 2)
                cv2.line(edges, (expected_x, self.sim_bed_y - 1), (expected_x + expected_w, self.sim_bed_y - 1), 255, 2)
        
        # 2. Edge Union
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        # 3. Contour isolation
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        deviation_pct = 0.0
        complexity_score = 0.0
        defect_bbox = None
        severity = "SAFE"
        
        active_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                _, _, w_box, _ = cv2.boundingRect(c)
                # Exclude ultra-wide horizontal structures (like the print bed line)
                if w_box < 300:
                    active_contours.append(c)
        
        if len(active_contours) > 0:
            active_contours = sorted(active_contours, key=lambda c: cv2.boundingRect(c)[1])
            xs = [cv2.boundingRect(c)[0] for c in active_contours]
            ys = [cv2.boundingRect(c)[1] for c in active_contours]
            ws = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in active_contours]
            hs = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in active_contours]
            
            gx, gy = min(xs), min(ys)
            gw, gh = max(ws) - gx, max(hs) - gy
            defect_bbox = (gx, gy, gw, gh)
            
            # --- Auto-calibration: Compare base width vs printing top width ---
            base_contours = [c for c in active_contours if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= (gy + gh - 60)]
            
            if base_contours:
                self.nominal_width = float(np.median([cv2.boundingRect(c)[2] for c in base_contours]))
                self.nominal_width = max(15.0, self.nominal_width)
                
            # --- Active G-Code / CAD Blueprint Synchronization (Ghost Box) ---
            if self.sim_layer > 0:
                # In simulator mode: expected perfect vertical cylinder
                expected_w = self.sim_pillar_width
                expected_x = self.sim_pillar_x - expected_w // 2
                expected_y = self.sim_nozzle_y
                expected_h = self.sim_bed_y - self.sim_nozzle_y
            else:
                # In real video mode: center Ghost Box around base contour's median X coordinate
                expected_w = max(15.0, self.nominal_width)
                if base_contours:
                    base_x_center = np.median([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2.0 for c in base_contours])
                else:
                    base_x_center = gx + gw / 2.0
                expected_x = int(base_x_center - expected_w / 2.0)
                expected_y = gy
                expected_h = gh

            # Bound CAD Ghost Box within the frame boundaries
            expected_h = max(5, expected_h)
            expected_x = max(0, min(self.width - 5, int(expected_x)))
            expected_y = max(0, min(self.height - 5, int(expected_y)))
            expected_w = max(5, min(self.width - expected_x, int(expected_w)))
            expected_h = max(5, min(self.height - expected_y, int(expected_h)))

            # 1. Create a blank mask for the expected CAD blueprint (The Ghost Box)
            cad_mask = np.zeros_like(gray)
            cv2.rectangle(cad_mask, (expected_x, expected_y), (expected_x + expected_w, expected_y + expected_h), 255, -1)

            # 2. Create a mask for the actual physical plastic detected by the camera
            actual_mask = np.zeros_like(gray)
            for c in active_contours:
                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                cv2.rectangle(actual_mask, (x_b, y_b), (x_b + w_b, y_b + h_b), 255, -1)

            # 3. Calculate the intersection (where physical plastic matches the CAD)
            intersection = cv2.bitwise_and(cad_mask, actual_mask)
            intersection_area = cv2.countNonZero(intersection)
            actual_area = cv2.countNonZero(actual_mask)

            # 4. Calculate Structural Deviation via Intersection over Union (IoU) ratio plummets
            if actual_area > 0:
                # How much of the physical plastic is OUTSIDE the expected CAD boundaries?
                rouge_plastic_ratio = (actual_area - intersection_area) / actual_area
                deviation_pct = rouge_plastic_ratio * 100.0
            else:
                deviation_pct = 0.0

            deviation_pct = min(100.0, max(0.0, deviation_pct))
            
            # --- Contour Complexity (Compactness) ---
            total_perimeter = sum(cv2.arcLength(c, True) for c in active_contours)
            total_area = sum(cv2.contourArea(c) for c in active_contours)
            if total_area > 0:
                compactness = (total_perimeter ** 2) / (4 * np.pi * total_area)
                complexity_score = min(100.0, max(0.0, (compactness - 1.0) * 8.0))
                
            # ---------------- NEW ANTI-JITTER FILTER ----------------
            # 1. Store the raw deviation in our history queue
            self.deviation_history.append(deviation_pct)
            
            # 2. Keep only the last 5 frames of history
            if len(self.deviation_history) > 5:
                self.deviation_history.pop(0)
                
            # 3. Calculate the smooth average
            smoothed_deviation = sum(self.deviation_history) / len(self.deviation_history)
            
            # 4. Fuse the smooth metrics into the final dynamic score
            defect_score = (smoothed_deviation * 0.6) + (complexity_score * 0.4)
            deviation_pct = min(100.0, max(smoothed_deviation, defect_score))
            
            if self.defect_active and self.anomaly_mode == "Under-Extrusion / Nozzle Clog":
                deviation_pct = self.defect_severity * 65.0
                complexity_score = self.defect_severity * 45.0
            # --------------------------------------------------------
            
            # --- Active ML SVM Classifier Inference ---
            if self.clf is not None:
                try:
                    contour_count = float(len(contours))
                    edge_density = float(np.count_nonzero(edges)) / (self.width * self.height)
                    areas = [float(cv2.contourArea(c)) for c in active_contours if cv2.contourArea(c) > 5]
                    mean_area = float(np.mean(areas)) if areas else 0.0
                    std_area = float(np.std(areas)) if areas else 0.0
                    
                    feat_vector = np.array([[
                        deviation_pct,
                        complexity_score,
                        contour_count,
                        edge_density,
                        mean_area,
                        std_area
                    ]], dtype=np.float32)
                    
                    pred = self.clf.predict(feat_vector)[0]
                    if hasattr(self.clf, "predict_proba"):
                        prob = self.clf.predict_proba(feat_vector)[0][1]
                    else:
                        prob = 1.0 if pred == 1 else 0.0
                        
                    self.ml_prediction = "DEFECT" if pred == 1 else "NOMINAL"
                    self.ml_probability = prob
                    
                    # Hybrid Safety System: if ML model predicts defect, ensure at least WARNING level is raised
                    if pred == 1 and deviation_pct < self.warning_threshold:
                        deviation_pct = max(deviation_pct, self.warning_threshold + 1.0)
                except Exception as e:
                    self.ml_prediction = "INFERENCE ERR"
                    self.ml_probability = 0.0
            else:
                self.ml_prediction = "HEURISTIC"
                self.ml_probability = 1.0 if (deviation_pct >= self.warning_threshold) else 0.0
            # ------------------------------------------

            if deviation_pct >= self.critical_threshold:
                severity = "CRITICAL"
                color = (0, 0, 220)
            elif deviation_pct >= self.warning_threshold:
                severity = "WARNING"
                color = (0, 140, 255)
            else:
                severity = "SAFE"
                color = (0, 180, 80)
                
            # Overlays
            # Draw Holographic transparent G-Code Ghost Box CAD layer
            overlay = processed.copy()
            cv2.rectangle(overlay, (expected_x, expected_y), (expected_x + expected_w, expected_y + expected_h), (255, 100, 0), -1) # Faint blue fill
            cv2.addWeighted(overlay, 0.15, processed, 0.85, 0, processed)
            
            # Faint electric blue border outline
            cv2.rectangle(processed, (expected_x, expected_y), (expected_x + expected_w, expected_y + expected_h), (255, 150, 0), 1, cv2.LINE_AA)
            cv2.putText(processed, "CAD G-CODE SYNC", (expected_x + 2, expected_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 0), 1, cv2.LINE_AA)

            # Draw active contours and actual bounding box
            for c in active_contours:
                cv2.drawContours(processed, [c], -1, color, 1)
            cv2.rectangle(processed, (gx, gy), (gx + gw, gy + gh), color, 2)
            
            # Render Live RBF-SVM prediction overlay (Cyan/Blue for NOMINAL, solid RED for DEFECT)
            ml_color = (255, 255, 0) if self.ml_prediction == "NOMINAL" else (0, 0, 255)
            cv2.putText(processed, f"SVM v4.0: {self.ml_prediction} ({self.ml_probability*100:.1f}%)", (gx, max(20, gy - 42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, ml_color, 1, cv2.LINE_AA)
            
            cv2.putText(processed, f"DEV: {deviation_pct/100.0:.3f}", (gx, max(20, gy - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(processed, f"STATUS: {severity}", (gx, max(20, gy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.line(processed, (self.width // 2, 0), (self.width // 2, self.height), (255, 255, 255), 1)
        else:
            severity = "SAFE"
            deviation_pct = 0.0
            complexity_score = 0.0
            self.ml_prediction = "NOMINAL"
            self.ml_probability = 0.0
            
        # Draw Camera Region of Interest (ROI) if enabled
        if self.roi_enabled:
            roi_color = (0, 220, 220) # Tech yellow
            cv2.rectangle(processed, (self.roi_x_min, self.roi_y_min), (self.roi_x_max, self.roi_y_max), roi_color, 1, cv2.LINE_AA)
            cv2.putText(processed, "CAMERA ROI ACTIVE", (self.roi_x_min + 5, self.roi_y_min + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, roi_color, 1, cv2.LINE_AA)

        return processed, {
            "deviation_pct": round(deviation_pct, 2),
            "complexity_score": round(complexity_score, 2),
            "severity_level": severity,
            "detected_score": round(deviation_pct / 100.0, 3),
            "bounding_box": defect_bbox,
            "ml_prediction": self.ml_prediction,
            "ml_probability": round(self.ml_probability, 4),
            "model_version": self.model_version
        }
