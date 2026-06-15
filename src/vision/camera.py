import cv2
import numpy as np
import threading
import time
import math
import random
from config import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX

class CameraManager:
    """
    Thread-safe camera manager that handles real camera streams
    and generates a high-fidelity 3D printer procedural anomaly simulator.
    """
    def __init__(self, camera_index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, simulated=True):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        # Source modes: "simulator", "webcam", "video"
        self.source_mode = "simulator" if simulated else "webcam"
        self.video_path = "data/sample_fail.mp4"
        
        # Thread control
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Latest frame buffer
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Camera & Video capture objects
        self.cap = None
        self.video_cap = None
        
        # --- Simulator State variables ---
        self.sim_layer = 0
        self.sim_max_layers = 300
        self.sim_bed_y = self.height - 80
        self.sim_pillar_x = self.width // 2
        self.sim_pillar_width = 40
        self.sim_nozzle_x = self.width // 2
        self.sim_nozzle_y = self.sim_bed_y
        
        # Simulation options
        self.defect_active = False
        self.defect_severity = 0.0  # 0.0 to 1.0
        self.spaghetti_points = []   # List of random curves for spaghetti effect
        self.sim_layer_lines = []    # History of printed solid layers
        self.print_speed = 0.8       # How fast the height grows
        self.time_elapsed = 0.0
        
        self.reset_simulation()

    @property
    def simulated(self):
        """Deprecated boolean check - maps to simulator source mode."""
        return self.source_mode == "simulator"

    def reset_simulation(self):
        """Resets the simulation to the beginning of the print."""
        with self.lock:
            self.sim_layer = 0
            self.sim_nozzle_y = self.sim_bed_y
            self.sim_layer_lines = []
            self.spaghetti_points = []
            self.defect_active = False
            self.defect_severity = 0.0
            self.time_elapsed = 0.0

    def set_source_mode(self, mode: str, video_path: str = None):
        """Sets active source mode ('simulator', 'webcam', 'video') and handles resource allocation."""
        with self.lock:
            if mode not in ["simulator", "webcam", "video"]:
                return
                
            self.source_mode = mode
            if video_path is not None:
                self.video_path = video_path
                
            # Release resources
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
                
            # Re-initialize depending on selection
            if self.source_mode == "webcam" and self.running:
                self._init_physical_camera()
            elif self.source_mode == "video" and self.running:
                self._init_video_camera()

    def set_simulated(self, state: bool):
        """Maintains backward compatibility with boolean toggling."""
        self.set_source_mode("simulator" if state else "webcam")

    def trigger_defect(self, state: bool):
        """Manually enables or disables the spaghetti effect in the simulator."""
        with self.lock:
            self.defect_active = state
            if not state:
                self.defect_severity = 0.0
                self.spaghetti_points = []

    def start(self):
        """Starts the background frame capture/generation thread."""
        if self.running:
            return
        
        self.running = True
        if self.source_mode == "webcam":
            self._init_physical_camera()
        elif self.source_mode == "video":
            self._init_video_camera()
            
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the background thread and releases resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None

    def read(self):
        """Thread-safe read of the latest frame."""
        with self.lock:
            return self.frame.copy()

    def _init_physical_camera(self):
        """Initializes the physical webcam capture device."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            self.source_mode = "simulator"

    def _init_video_camera(self):
        """Initializes video file capture stream."""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
        except Exception:
            self.source_mode = "simulator"

    def _capture_loop(self):
        """Background thread main loop running at ~30 FPS."""
        frame_delay = 1.0 / 30.0  # ~33 ms
        
        while self.running:
            start_time = time.time()
            
            if self.source_mode == "simulator":
                frame = self._generate_simulated_frame()
            elif self.source_mode == "webcam":
                frame = self._read_physical_camera()
            elif self.source_mode == "video":
                frame = self._read_video_frame()
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
            with self.lock:
                self.frame = frame
                
            # Maintain stable framerate
            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)

    def _read_physical_camera(self):
        """Captures a frame from the live webcam feed."""
        if self.cap is None or not self.cap.isOpened():
            # If camera isn't ready, return a black warning screen
            return self._create_warning_frame("CAMERA DEVICE NOT READY")
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return self._create_warning_frame("FAILED TO GRAB FRAME")
            
        # Resize to standard workspace size
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        return frame

    def _read_video_frame(self):
        """Captures a frame from a local video file, looping automatically."""
        if self.video_cap is None or not self.video_cap.isOpened():
            return self._create_warning_frame("VIDEO FILE NOT ACCESSIBLE")
            
        ret, frame = self.video_cap.read()
        if not ret or frame is None:
            # Loop the video file back to start frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if not ret or frame is None:
                return self._create_warning_frame("VIDEO FILE PLAYBACK COMPLETED")
                
        # Resize to fit the pipeline workspace
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        return frame

    def _create_warning_frame(self, message):
        """Helper to create a diagnostic screen if hardware fails."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Deep industrial grid
        self._draw_background_grid(frame)
        cv2.putText(frame, "HARDWARE OFFLINE", (self.width // 2 - 140, self.height // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, message, (self.width // 2 - 120, self.height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return frame

    def _draw_background_grid(self, frame):
        """Draws a sleek cyber-industrial 3D space grid background."""
        frame[:] = (20, 22, 28) # Clean dark metallic blue-grey
        
        # Grid settings
        grid_color = (35, 38, 48)
        grid_step = 40
        
        # Draw vertical lines
        for x in range(0, self.width, grid_step):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)
        # Draw horizontal lines
        for y in range(0, self.height, grid_step):
            cv2.line(frame, (0, y), (self.width, y), grid_color, 1)

    def _generate_simulated_frame(self):
        """
        Generates a visually stunning, high-fidelity 3D printing simulation.
        Can transition from printing a perfect vertical pillar to generating a
        chaotic 'spaghetti effect' structural failure.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_background_grid(frame)
        
        # Draw localized coordinate info
        cv2.putText(frame, "EDGE TWIN SIMULATOR: ON", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"LAYER: {int(self.sim_layer)} / {self.sim_max_layers}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 130, 140), 1, cv2.LINE_AA)
        
        # Update simulation variables
        self.time_elapsed += 0.033 # Approx time step for 30 FPS
        
        if self.sim_layer < self.sim_max_layers:
            # Gradually print upwards
            self.sim_layer += self.print_speed
            self.sim_nozzle_y = self.sim_bed_y - int(self.sim_layer * 1.0)
            
            # Print head moves side-to-side laying down filament
            self.sim_nozzle_x = self.sim_pillar_x + int(math.sin(self.time_elapsed * 10) * (self.sim_pillar_width // 2))
            
            # Save normal printed coordinate history
            if not self.defect_active:
                # Store structural horizontal layer lines
                if int(self.sim_layer) % 2 == 0:
                    y_coord = int(self.sim_nozzle_y)
                    self.sim_layer_lines.append((self.sim_pillar_x, y_coord, self.sim_pillar_width))
            else:
                # Ramp defect severity
                self.defect_severity = min(1.0, self.defect_severity + 0.005)
        
        # --- DRAW PRINTED GEOMETRY ---
        # 1. Draw printed solid foundation (the healthy pillar built so far)
        for pillar_x, y_coord, width in self.sim_layer_lines:
            half_w = width // 2
            # Safe digital green layering
            layer_color = (0, 180, 80)
            cv2.line(frame, (pillar_x - half_w, y_coord), (pillar_x + half_w, y_coord), layer_color, 2)
            
        # 2. Draw Simulated Anomaly (Spaghetti Effect) if active
        if self.defect_active and len(self.sim_layer_lines) > 0:
            # Get the top of the healthy printed pillar
            top_healthy_y = self.sim_layer_lines[-1][1] if self.sim_layer_lines else self.sim_bed_y
            
            # Generate new spaghetti points originating around nozzle/top
            if len(self.spaghetti_points) < 80:
                # Add random walks / chaotic lines starting from printer head
                start_pt = (self.sim_nozzle_x, self.sim_nozzle_y)
                current_curve = [start_pt]
                
                cx, cy = start_pt
                # Generate a chaotic squiggly filament chain
                for _ in range(random.randint(5, 15)):
                    # Defect spreads outwards and drops due to gravity
                    cx += random.randint(-20, 20)
                    cy += random.randint(-5, 25) # Gravitational droop
                    # Clamp inside boundaries
                    cx = max(10, min(self.width - 10, cx))
                    cy = max(top_healthy_y - 20, min(self.sim_bed_y, cy))
                    current_curve.append((cx, cy))
                
                self.spaghetti_points.append(current_curve)
                
            # Render spaghetti lines with physical failure aesthetics (chaotic plastic extrusion)
            for curve in self.spaghetti_points[:int(len(self.spaghetti_points) * self.defect_severity)]:
                for i in range(len(curve) - 1):
                    # Pulsing hot warning colors (Amber to Crimson)
                    p1 = curve[i]
                    p2 = curve[i+1]
                    # Dynamic color based on severity
                    color = (0, int(100 + 155 * (1 - self.defect_severity)), int(180 + 75 * self.defect_severity)) # Greenish to Orange-Red
                    cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        # 3. Draw Build Plate
        cv2.line(frame, (50, self.sim_bed_y), (self.width - 50, self.sim_bed_y), (100, 105, 115), 3)
        cv2.putText(frame, "BUILD PLATE", (55, self.sim_bed_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 105, 115), 1, cv2.LINE_AA)

        # 4. Draw Simulated Printer Head (Gantry and Nozzle)
        if self.sim_layer < self.sim_max_layers:
            ny = self.sim_nozzle_y
            nx = self.sim_nozzle_x
            
            # Draw gantry horizontal bar
            cv2.line(frame, (0, ny - 15), (self.width, ny - 15), (60, 65, 75), 2)
            
            # Draw extruder assembly block
            cv2.rectangle(frame, (nx - 20, ny - 30), (nx + 20, ny - 10), (80, 85, 95), -1)
            cv2.rectangle(frame, (nx - 20, ny - 30), (nx + 20, ny - 10), (120, 125, 135), 1)
            
            # Draw hotend glowing nozzle (Yellow/Red glow indicating extrusion heat)
            nozzle_color = (0, 120, 255) if not self.defect_active else (0, 0, 255)
            # Tri-nozzle tip
            pts = np.array([[nx - 6, ny - 10], [nx + 6, ny - 10], [nx, ny]], np.int32)
            cv2.fillPoly(frame, [pts], nozzle_color)
            
            # Extrusion glow dot
            cv2.circle(frame, (nx, ny), 3, (180, 220, 255), -1)
            
        return frame
