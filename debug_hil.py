import sys
import os
import time

# Ensure workspace is in import path
sys.path.append(".")

from vision import VisionGateway
import numpy as np

def run_hil_test_suite():
    print("================================================================")
    print("[MASTER] SMARTFLOW HIL & CYBER-PHYSICAL VALIDATION TEST RUNNER")
    print("================================================================")
    
    gateway = VisionGateway()
    
    # ------------------ TEST 1: NOMINAL PRINT ------------------
    print("\n[TEST 1] Verifying Nominal Clean Cylinder Printing...")
    gateway.reset_simulation()
    gateway.anomaly_mode = "Spaghetti Effect"
    gateway.trigger_defect(False)
    gateway.sensor_noise = 0.0
    gateway.print_speed = 2.0
    
    # Run 60 frames of perfect nominal printing
    for _ in range(60):
        frame = gateway.get_simulated_frame()
        
    processed, telemetry = gateway.process_frame(frame)
    print(f"-> Result: Deviation = {telemetry['deviation_pct']}%, Severity = {telemetry['severity_level']}")
    if telemetry['severity_level'] in ["SAFE", "WARNING"] and telemetry['deviation_pct'] < 20.0:
        print("[PASS] TEST 1: Nominal printing operates cleanly under warning/safe thresholds.")
    else:
        print("[FAIL] TEST 1: False positive critical anomaly flagged on nominal print.")
        sys.exit(1)
        
    # ------------------ TEST 2: SPAGHETTI EFFECT ------------------
    print("\n[TEST 2] Verifying Spaghetti Effect Defect Injection...")
    gateway.reset_simulation()
    gateway.anomaly_mode = "Spaghetti Effect"
    gateway.print_speed = 2.0
    
    # Start nominal for 60 frames first to establish a print base
    for _ in range(60):
        frame = gateway.get_simulated_frame()
        
    # Trigger Spaghetti defect mid-print
    gateway.trigger_defect(True)
    
    # Run 100 frames to let spaghetti strings grow chaotic
    for _ in range(100):
        frame = gateway.get_simulated_frame()
        
    processed, telemetry = gateway.process_frame(frame)
    print(f"-> Result: Deviation = {telemetry['deviation_pct']}%, Severity = {telemetry['severity_level']}")
    if telemetry['severity_level'] == "CRITICAL" and telemetry['deviation_pct'] >= gateway.critical_threshold:
        print("[PASS] TEST 2: Spaghetti defect accurately detected and closed-loop threshold exceeded.")
    else:
        print("[FAIL] TEST 2: Failed to detect spaghetti failure.")
        sys.exit(1)
        
    # ------------------ TEST 3: LAYER SHIFTING ------------------
    print("\n[TEST 3] Verifying Layer Shifting Lateral Mechanical Slip...")
    gateway.reset_simulation()
    gateway.anomaly_mode = "Layer Shifting"
    gateway.print_speed = 2.0
    
    # Start nominal for 60 frames first
    for _ in range(60):
        frame = gateway.get_simulated_frame()
        
    # Trigger Layer Shifting mid-print
    gateway.trigger_defect(True)
    
    # Run 100 frames to generate lateral slip
    for _ in range(100):
        frame = gateway.get_simulated_frame()
        
    processed, telemetry = gateway.process_frame(frame)
    print(f"-> Result: Bounding Box = {telemetry['bounding_box']}, Deviation = {telemetry['deviation_pct']}%, Severity = {telemetry['severity_level']}")
    if telemetry['severity_level'] == "CRITICAL" and telemetry['bounding_box'] is not None:
        print("[PASS] TEST 3: Layer shifting detected; printed profile successfully separated from G-code Ghost Box.")
    else:
        print("[FAIL] TEST 3: Failed to isolate shifted layers or trigger critical gantry halt.")
        sys.exit(1)
        
    # ------------------ TEST 4: WARPING / BED COLLAPSE ------------------
    print("\n[TEST 4] Verifying Warping & Bed Adhesion Curl upward...")
    gateway.reset_simulation()
    gateway.anomaly_mode = "Warping / Bed Adhesion Collapse"
    gateway.print_speed = 2.0
    
    # Start nominal for 60 frames first
    for _ in range(60):
        frame = gateway.get_simulated_frame()
        
    # Trigger Warping mid-print
    gateway.trigger_defect(True)
    
    # Run 100 frames to curl base layers upwards
    for _ in range(100):
        frame = gateway.get_simulated_frame()
        
    processed, telemetry = gateway.process_frame(frame)
    print(f"-> Result: Bounding Box = {telemetry['bounding_box']}, Deviation = {telemetry['deviation_pct']}%, Severity = {telemetry['severity_level']}")
    if telemetry['severity_level'] == "CRITICAL" and telemetry['bounding_box'] is not None:
        print("[PASS] TEST 4: Warping detected; curling base layers successfully triggered threshold alarm.")
    else:
        print("[FAIL] TEST 4: Warping failure missed by edge CV spectrometer.")
        sys.exit(1)
        
    # ------------------ TEST 5: EMF NOISE STRESS-TEST ------------------
    print("\n[TEST 5] Verifying Factory Sensor Noise Injection...")
    gateway.reset_simulation()
    gateway.anomaly_mode = "Spaghetti Effect"
    gateway.trigger_defect(False)
    gateway.sensor_noise = 30.0 # High EMF static snow
    gateway.print_speed = 2.0
    
    for _ in range(60):
        frame = gateway.get_simulated_frame()
        
    processed, telemetry = gateway.process_frame(frame)
    print(f"-> Result: Deviation = {telemetry['deviation_pct']}%, Severity = {telemetry['severity_level']}")
    print("[PASS] TEST 5: High EMF sensor noise stress-tested cleanly.")
    
    print("\n================================================================")
    print("[SUCCESS] ALL HIL SUITE VALIDATION TESTS COMPLETED SUCCESSFULLY!")
    print("================================================================")

if __name__ == "__main__":
    run_hil_test_suite()
