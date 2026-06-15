import cv2
import numpy as np
from config import THRESHOLD_WARNING, THRESHOLD_CRITICAL, FRAME_WIDTH, FRAME_HEIGHT

class DefectDetector:
    """
    Computer Vision analyzer that processes 3D printing frames.
    Uses contour analysis, complexity measurements, and aspect ratio tracking
    to detect structural deviations like the 'spaghetti effect' in real-time.
    """
    def __init__(self, warning_threshold=THRESHOLD_WARNING, critical_threshold=THRESHOLD_CRITICAL):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.nominal_width = 40.0  # Nominal width of the healthy vertical vertical cylinder/pillar
        self.historical_deviations = []
        self.deviation_history = []
        
        # Dynamic configuration sliders (defaults from user settings)
        self.canny_lower = 100
        self.complexity_threshold = 50.0

    def analyze_frame(self, frame, canny_lower=None, complexity_thresh=None, warning_thresh=None, critical_thresh=None):
        """
        Processes real-world video frames using a robust Canny-edge based self-calibrating pipeline.
        Designed to generalize across varying lighting conditions, filament colors, and camera zooms
        typical of open-source Kaggle and Roboflow print defect datasets.
        """
        # Apply live slider updates from Control Deck
        if canny_lower is not None:
            self.canny_lower = canny_lower
        if complexity_thresh is not None:
            self.complexity_threshold = complexity_thresh
        if warning_thresh is not None:
            self.warning_threshold = warning_thresh
        if critical_thresh is not None:
            self.critical_threshold = critical_thresh
            
        processed_frame = frame.copy()
        
        # 1. Image Pre-processing
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to smooth video compression noise and camera artifacts
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Lighting-Robust Edge Segmentation (Canny)
        # Instead of static thresholding, Canny isolates high-frequency filament borders 
        # using the dynamically set user threshold.
        upper_canny = min(255, int(self.canny_lower * 1.5))
        edges = cv2.Canny(blurred, self.canny_lower, upper_canny)
        
        # 3. Morphological Union
        # Dilate and close edges to bridge micro-gaps between chaotic spaghetti strands
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        # Find outer contours of all printed items
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Telemetry metrics
        deviation_pct = 0.0
        complexity_score = 0.0
        defect_bounding_box = None
        severity_level = "SAFE"
        contour_count = len(contours)
        
        active_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter background dust, grid lines, or mechanical rails below 200px area
            if area > 200:
                _, _, w_box, _ = cv2.boundingRect(c)
                # Exclude ultra-wide horizontal structures (like the print bed line)
                if w_box < 300:
                    active_contours.append(c)
                
        if len(active_contours) > 0:
            # Sort active contours vertically (top-most Y coordinate first)
            active_contours = sorted(active_contours, key=lambda c: cv2.boundingRect(c)[1])
            
            # Identify physical print bed base anchor vs. active printing envelope
            xs, ys, ws, hs = [], [], [], []
            for c in active_contours:
                x, y, w, h = cv2.boundingRect(c)
                xs.append(x)
                ys.append(y)
                ws.append(x + w)
                hs.append(y + h)
                
            global_x = min(xs)
            global_y = min(ys)
            global_w = max(ws) - global_x
            global_h = max(hs) - global_y
            
            defect_bounding_box = (global_x, global_y, global_w, global_h)
            
            # --- AUTO-CALIBRATING PHYSICAL DEVIATION ALGORITHM ---
            # Instead of assuming the healthy pillar is 40px wide, we dynamically measure the 
            # anchor width at the base (print bed bottom 20% slice) where the print is stable, 
            # and compare it to the nozzle printing envelope (top 20% slice).
            
            # Base anchor calculation
            base_contours = [c for c in active_contours if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= (global_y + global_h - 60)]
            top_contours = [c for c in active_contours if cv2.boundingRect(c)[1] <= (global_y + 60)]
            
            # Baseline pillar width calibration
            if base_contours:
                base_w_list = [cv2.boundingRect(c)[2] for c in base_contours]
                self.nominal_width = float(np.median(base_w_list))
                # Protect against zero division or tiny widths
                self.nominal_width = max(15.0, self.nominal_width)
            
            # Printing envelope current width
            if top_contours:
                top_xs = [cv2.boundingRect(c)[0] for c in top_contours]
                top_xmaxs = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in top_contours]
                current_top_width = max(top_xmaxs) - min(top_xs)
            else:
                current_top_width = global_w
                
            # Structural deviation percentage relative to calibrated baseline
            if current_top_width > self.nominal_width:
                deviation_pct = ((current_top_width - self.nominal_width) / self.nominal_width) * 100.0
            else:
                deviation_pct = 0.0
                
            deviation_pct = min(100.0, deviation_pct)
            
            # --- CONTOUR COMPLEXITY INDEX (Filament Chaos Tracking) ---
            total_perimeter = 0.0
            total_area = 0.0
            for c in active_contours:
                total_perimeter += cv2.arcLength(c, True)
                total_area += cv2.contourArea(c)
                
            if total_area > 0:
                # Isoperimetric compactness ratio
                compactness = (total_perimeter ** 2) / (4 * np.pi * total_area)
                # Normalize ratio to a logical score (0 to 100 index)
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
            defect_intensity = (smoothed_deviation * 0.6) + (complexity_score * 0.4)
            deviation_pct = min(100.0, max(smoothed_deviation, defect_intensity))
            # --------------------------------------------------------
            
            # safety classification transitions
            if deviation_pct >= self.critical_threshold:
                severity_level = "CRITICAL"
                hud_color = (0, 0, 220)  # Pulse Crimson
            elif deviation_pct >= self.warning_threshold:
                severity_level = "WARNING"
                hud_color = (0, 140, 255) # Warning Amber
            else:
                severity_level = "SAFE"
                hud_color = (0, 180, 80)   # Healthy Green
                
            # --- RENDER COMPUTER VISION HUD OVERLAYS ---
            # Highlight filament boundaries
            for c in active_contours:
                cv2.drawContours(processed_frame, [c], -1, hud_color, 1)
                
            # Draw tracking envelope box
            cv2.rectangle(processed_frame, (global_x, global_y), (global_x + global_w, global_y + global_h), hud_color, 2)
            
            # HUD telemetry labels
            cv2.putText(processed_frame, f"DEV: {deviation_pct:.1f}%", (global_x, max(20, global_y - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)
            cv2.putText(processed_frame, f"STATUS: {severity_level}", (global_x, max(20, global_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)
            
            # Central symmetry guide line
            cv2.line(processed_frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (255, 255, 255), 1, cv2.LINE_4)
            
        else:
            severity_level = "SAFE"
            deviation_pct = 0.0
            complexity_score = 0.0
            
        # Smooth metrics temporal history
        self.historical_deviations.append(deviation_pct)
        if len(self.historical_deviations) > 100:
            self.historical_deviations.pop(0)
            
        telemetry = {
            "deviation_pct": round(deviation_pct, 2),
            "complexity_score": round(complexity_score, 2),
            "active_defect_detected": deviation_pct >= self.warning_threshold,
            "severity_level": severity_level,
            "contour_count": contour_count,
            "bounding_box": defect_bounding_box,
            "average_deviation_5s": round(np.mean(self.historical_deviations[-15:]), 2) if self.historical_deviations else 0.0
        }
        
        return processed_frame, telemetry
