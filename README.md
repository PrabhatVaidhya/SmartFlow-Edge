# SmartFlow: Edge AI Digital Twin (IIT KGP Silver Jubilee - 2 Crore Pitch Guide)

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

*Note: SmartFlow features a zero-downtime Local Heuristic Cognitive Engine. If Ollama is not installed or running, the dashboard automatically falls back to local edge reasoning with <150ms latency, guaranteeing a bulletproof, crash-free presentation on stage.*

---

## 💻 Running the Command Center Dashboard

To launch the real-time Streamlit digital twin dashboard, execute:
```bash
streamlit run app.py
```
A premium, dark-cybernetic console will automatically open in your default browser at `http://localhost:8501`.

---

## 🎯 Winning Pitch Presentation Walkthrough

Here is the exact step-by-step narrative to deliver an outstanding live demonstration for the **IIT KGP 2 Crore Competition**:

### Step 1: Establish the Baseline (Nominal Print)
1. **Launch the Dashboard:** Point out the dark-mode cyber gantry layout, glowing telemetry dials, and empty log stream.
2. **Nominal State:** The simulator begins printing a clean, vertical cylinder layer-by-layer. Point to the green color profile in the live feed.
3. **Telemetry & HUD:** Highlight the metrics under **Geometric Deviation (0.0%)**, **Extruder Temp (210°C)**, and **Feed Velocity (100%)**. Mention that the local OpenCV framework is analyzing the contour aspects in real-time.
4. **Explain Edge Privacy:** Emphasize to the judges that all image segmentation and model inferencing are happening completely locally on the device with zero cloud transmission.

### Step 2: Triggering the Anomaly (The Failure State)
1. **Trigger Defect:** Click the **`🔴 TRIGGER SPAGHETTI EFFECT`** button in the Control Deck.
2. **Defect Simulation:** The printer nozzle begins vibrating, extruding random, chaotic bezier filament loops dropping outwards.
3. **OpenCV Detection:** Point out that the CV engine immediately detects the chaotic structures, tracing them with glowing orange outlines and fitting a dynamic bounding box.
4. **Warning Stage (Aspect Ratio Spike):** The **Geometric Deviation** metric climbs past 15%. Explain that the CV pipeline has identified contour entropy.
5. **AI Reasoning:** The dashboard displays the **Edge AI Intelligence Layer** card updating. Ollama evaluates the deviation and generates a response:
   - **Mitigation Action:** `REDUCE FEED RATE & TEMP`
   - **Target G-Code:** `M220 S75; M104 S195`
   - Show how the physical console automatically responds by dimming the gantry speed and lowering hotend heat to stabilize the filament.

### Step 3: Closed-Loop Automation (Emergency Stop)
1. **Critical Defect Expansion:** As the spaghetti filament continues to expand, the **Geometric Deviation** crosses the critical threshold (35%).
2. **Closed-Loop Safety Trigger:** Point out that because **Closed-Loop AI Emergency Stop** is enabled, the system does not wait for a human operator.
3. **Emergency Halt:** The dashboard triggers `🚨 CLOSED-LOOP HALTED`. The printer feed velocity drops to 0%, the hotend heater is cut to 0°C, and the gantry freezes.
4. **Filament Savings:** Point to the **FILAMENT SAVED** display. Explain that early stoppage saved `45.2 grams` of high-performance polymer. Show how scale replication saves thousands of dollars in commercial operations.
5. **Log Integrity:** Point to the retro console box which shows the precise sequence of CV triggers, Ollama inferences, and G-code commands issued.

### Step 4: Demonstrate Algorithmic Robustness & Live Controls
1. **System Execution Mode:** Point to the top of the sidebar showing **`⚡ SYSTEM EXECUTION MODE`**. Explain that **Mode A: Cloud Evaluation Sandbox (Procedural Emulation)** is selected to run purely on internal mathematical loops and simulation inputs. Show that switching to **Mode B: Direct Fieldbus Edge Link (Physical Hardware)** activates our real-time `pyserial` connection pipelines to bind directly to local hardware serial ports.
2. **Chaos Testing (Stochastic Noise):** In the metrology deck, check **`💥 INTRODUCE STOCHASTIC SENSOR INTERFERENCE (HIGH COVARIANCE NOISE)`**. Point out how the raw charts spike with high-covariance noise, while the **Extended Kalman Filter (EKF)** indicator smoothly filters the signals, keeping the estimation clean.
3. **Master Reset:** When the judges ask to clear custom overrides, click the low-profile button at the absolute bottom of the sidebar: **`🔄 SYSTEM INITIALIZATION DETECTED // RESET TO FACTORY BASELINE`**. This wipes the session state to instantly restore a clean `STATE_NOMINAL` configuration.
4. **Conclusion:** End the pitch by highlighting that this modular stack (OpenCV + Ollama + Streamlit) represents a production-ready, edge-native industrial architecture capable of saving millions in manufacturing downtime.

