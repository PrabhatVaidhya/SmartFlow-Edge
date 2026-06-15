# SmartFlow Edge: AI Digital Twin

SmartFlow is an edge-native cyber-physical system designed to track, identify, and stop additive manufacturing failures (specifically the chaotic "spaghetti effect" on thin vertical geometries) in real-time. By utilizing localized computer vision combined with an Edge AI LLM, SmartFlow achieves instant closed-loop mitigation with zero cloud costs and total data privacy.

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your local system.

### 2. Clone & Install Dependencies
Navigate to the project root directory and install the required packages:
```bash
pip install -r requirements.txt
```

### 2.1 Generate Demo Assets (Optional)
If the sample anomaly footage is missing, build it locally with:
```bash
python generate_video_sample.py
```
This creates `data/sample_fail.mp4` so the dashboard always has a fallback demo video.

### 3. Setup Edge AI (Ollama)
SmartFlow runs local quantized models to perform advanced reasoning.
1. Download and install **Ollama** from [https://ollama.com](https://ollama.com).
2. Open your terminal and pull the standard high-performance reasoning model optimized for our cognitive thought process drawer:
   ```bash
   ollama pull deepseek-r1:1.5b
   ```
3. Keep the Ollama application running in the background.

*Note: SmartFlow features a zero-downtime Local Heuristic Cognitive Engine. If Ollama is not installed or running, the dashboard automatically falls back to local edge reasoning with <150ms latency, guaranteeing bulletproof, crash-free local fallback execution.*

---

## 💻 Running the Command Center Dashboard

To launch the real-time Streamlit digital twin dashboard, execute:
```bash
streamlit run app.py
```
A premium, dark-cybernetic console will automatically open in your default browser at `http://localhost:8501`.

---

## 🎯 System Demonstration & Walkthrough

Here is the step-by-step walkthrough to demonstrate the system's capabilities in real-time:

### Step 1: Establish the Baseline (Nominal Print)
1. **Launch the Dashboard:** Open the dashboard to see the dark-mode cyber gantry layout, glowing telemetry dials, and real-time log stream.
2. **Nominal State:** The simulator begins printing a clean, vertical cylinder layer-by-layer, indicated by a green color profile in the live feed.
3. **Telemetry & HUD:** Monitor the telemetry metrics under **Geometric Deviation (0.0%)**, **Extruder Temp (210°C)**, and **Feed Velocity (100%)**, driven by the local real-time OpenCV contour analysis.
4. **Edge Privacy:** All image segmentation and model inferencing happen completely locally on the device with zero cloud transmission.

### Step 2: Triggering the Anomaly (The Failure State)
1. **Trigger Defect:** Click the **`🔴 TRIGGER SPAGHETTI EFFECT`** button in the Control Deck.
2. **Defect Simulation:** The printer nozzle begins vibrating, extruding random, chaotic bezier filament loops dropping outwards.
3. **OpenCV Detection:** The computer vision engine immediately detects the chaotic structures, tracing them with glowing orange outlines and fitting a dynamic bounding box.
4. **Warning Stage (Aspect Ratio Spike):** The **Geometric Deviation** metric climbs past 15% as the CV pipeline identifies contour entropy.
5. **AI Reasoning:** The **Edge AI Intelligence Layer** card updates. Ollama evaluates the deviation and generates a response:
   - **Mitigation Action:** `REDUCE FEED RATE & TEMP`
   - **Target G-Code:** `M220 S75; M104 S195`
   - The system automatically responds by dimming the gantry speed and lowering hotend heat to stabilize the filament.

### Step 3: Closed-Loop Automation (Emergency Stop)
1. **Critical Defect Expansion:** As the spaghetti filament continues to expand, the **Geometric Deviation** crosses the critical threshold (35%).
2. **Closed-Loop Safety Trigger:** With the **Closed-Loop AI Emergency Stop** enabled, the system does not wait for a human operator.
3. **Emergency Halt:** The dashboard triggers `🚨 CLOSED-LOOP HALTED`. The printer feed velocity drops to 0%, the hotend heater is cut to 0°C, and the gantry freezes.
4. **Filament Savings:** The **FILAMENT SAVED** display shows how early stoppage saved `45.2 grams` of high-performance polymer.
5. **Log Integrity:** The console log outputs the precise sequence of CV triggers, Ollama inferences, and G-code commands issued.

### Step 4: Demonstrate Algorithmic Robustness & Live Controls
1. **System Execution Mode:** The sidebar displays **`⚡ SYSTEM EXECUTION MODE`**. **Mode A: Cloud Evaluation Sandbox (Procedural Emulation)** runs purely on internal mathematical loops and simulation inputs, while **Mode B: Direct Fieldbus Edge Link (Physical Hardware)** activates real-time `pyserial` connection pipelines to bind directly to local hardware serial ports.
2. **Chaos Testing (Stochastic Noise):** In the metrology deck, check **`💥 INTRODUCE STOCHASTIC SENSOR INTERFERENCE (HIGH COVARIANCE NOISE)`** to inject high-covariance noise. Observe how the **Extended Kalman Filter (EKF)** smoothly filters the signals, keeping the estimation clean.
3. **Master Reset:** Click the button at the bottom of the sidebar: **`🔄 SYSTEM INITIALIZATION DETECTED // RESET TO FACTORY BASELINE`** to wipe the session state and restore a clean `STATE_NOMINAL` configuration.
4. **Conclusion:** This modular stack (OpenCV + Ollama + Streamlit) represents a production-ready, edge-native industrial architecture capable of saving significant manufacturing downtime and materials.

