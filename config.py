import os

# --- Hardware & Camera Configuration ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# --- Anomaly Detection Parameters ---
# Thresholds for structural deviation in percentage
THRESHOLD_WARNING = 15.0      # Trigger warnings above 15% deviation
THRESHOLD_CRITICAL = 35.0     # Trigger automated shutdown above 35% deviation

# --- Local Intelligence (Ollama) Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Standard lightweight models optimized for fast edge inference on 8GB RAM systems
OLLAMA_MODEL = "deepseek-r1:1.5b"  # Alternatives: "tinyllama", "qwen2.5:0.5b", "tinydolphin"
OLLAMA_TIMEOUT = 5.0       # Seconds before fallback is triggered

# --- File System Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
SAVED_ANOMALIES_DIR = os.path.join(DATA_DIR, "saved_anomalies")

# Ensure critical data directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SAVED_ANOMALIES_DIR, exist_ok=True)

# Path to system logs
LOG_FILE_PATH = os.path.join(LOGS_DIR, "system_events.csv")
