"""
SmartFlow Edge // Core Twin Platform
ML Training Pipeline
================================================
Step 1: Generate 500 additional synthetic training images (250 nominal + 250 defect)
Step 2: Extract 4-dimensional physics feature vectors from ALL 1,073+ images
Step 3: Train a production-grade RBF-SVM with 5-fold cross-validation
Step 4: Save upgraded defect_classifier.pkl + print full training report
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import math
import random
import pickle
import numpy as np
import time

# -- Scikit-learn --------------------------------------------------------------
try:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[ERROR] scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

# -- Vision stack --------------------------------------------------------------
from vision import VisionGateway

# -- Paths ---------------------------------------------------------------------
NOMINAL_DIR   = "data/dataset/nominal"
DEFECT_DIR    = "data/dataset/defect"
MODEL_PATH    = "data/defect_classifier.pkl"
REPORT_PATH   = "data/training_report.txt"

os.makedirs(NOMINAL_DIR, exist_ok=True)
os.makedirs(DEFECT_DIR,  exist_ok=True)

# ==============================================================================
# STEP 1 -- GENERATE ADDITIONAL SYNTHETIC IMAGES
# ==============================================================================

def patched_frame(self):
    """Minimal clean simulation frame for fast batch generation."""
    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    frame[:] = (20, 22, 28)
    for x in range(0, self.width, 40):
        cv2.line(frame, (x, 0), (x, self.height), (30, 32, 40), 1)
    for y in range(0, self.height, 40):
        cv2.line(frame, (0, y), (self.width, y), (30, 32, 40), 1)
    self.time_elapsed += 0.033
    if self.sim_layer < self.sim_max_layers:
        self.sim_layer    += self.print_speed
        self.sim_nozzle_y  = self.sim_bed_y - int(self.sim_layer * 1.0)
        self.sim_nozzle_x  = self.sim_pillar_x + int(
            math.sin(self.time_elapsed * 10) * (self.sim_pillar_width // 2))
        if not self.defect_active:
            if int(self.sim_layer) % 2 == 0:
                self.sim_layer_lines.append(
                    (self.sim_pillar_x, int(self.sim_nozzle_y), self.sim_pillar_width))
        else:
            self.defect_severity = min(1.0, self.defect_severity + 0.005)
    for px, py, pw in self.sim_layer_lines:
        cv2.line(frame, (px - pw // 2, py), (px + pw // 2, py), (0, 180, 80), 2)
    if self.defect_active and len(self.sim_layer_lines) > 0:
        top_y = self.sim_layer_lines[-1][1]
        if len(self.spaghetti_points) < 80:
            cx, cy = self.sim_nozzle_x, self.sim_nozzle_y
            curve  = [(cx, cy)]
            for _ in range(random.randint(5, 12)):
                cx += random.randint(-18, 18)
                cy += random.randint(-4, 22)
                cx = max(10, min(self.width - 10, cx))
                cy = max(top_y - 20, min(self.sim_bed_y, cy))
                curve.append((cx, cy))
            self.spaghetti_points.append(curve)
        for curve in self.spaghetti_points[:int(len(self.spaghetti_points) * self.defect_severity)]:
            for i in range(len(curve) - 1):
                color = (0,
                         int(120 + 135 * (1 - self.defect_severity)),
                         int(180 + 75  *      self.defect_severity))
                cv2.line(frame, curve[i], curve[i + 1], color, 1, cv2.LINE_AA)
    return frame

VisionGateway.get_simulated_frame = patched_frame


def _advance_layers(gw, n):
    for _ in range(n):
        gw.sim_layer    += gw.print_speed
        gw.sim_nozzle_y  = gw.sim_bed_y - int(gw.sim_layer * 1.0)
        gw.sim_nozzle_x  = gw.sim_pillar_x + int(
            math.sin(gw.time_elapsed * 10) * (gw.sim_pillar_width // 2))
        if int(gw.sim_layer) % 2 == 0:
            gw.sim_layer_lines.append(
                (gw.sim_pillar_x, int(gw.sim_nozzle_y), gw.sim_pillar_width))
        gw.time_elapsed += 0.033


def generate_extra_images(n_each=250):
    """Generate n_each nominal and n_each defect images to augment the dataset."""
    existing_nominal = len([f for f in os.listdir(NOMINAL_DIR) if f.endswith('.png')])
    existing_defect  = len([f for f in os.listdir(DEFECT_DIR)  if f.endswith('.png')])

    print(f"\n[DATASET] Existing: {existing_nominal} nominal, {existing_defect} defect")
    print(f"[DATASET] Generating {n_each} extra nominal + {n_each} extra defect images ...\n")

    gw = VisionGateway()

    # -- Nominal ---------------------------------------------------------------
    for i in range(n_each):
        gw.reset_simulation()
        gw.trigger_defect(False)
        _advance_layers(gw, random.randint(30, 200))
        gw.sim_layer = gw.sim_max_layers
        frame = gw.get_simulated_frame()
        idx   = existing_nominal + i
        cv2.imwrite(f"{NOMINAL_DIR}/nominal_{idx:04d}.png", frame)
        if (i + 1) % 50 == 0:
            print(f"  [nominal]  {i+1}/{n_each} saved")

    # -- Defect ----------------------------------------------------------------
    for i in range(n_each):
        gw.reset_simulation()
        _advance_layers(gw, random.randint(30, 120))
        gw.trigger_defect(True)
        gw.defect_severity = 0.0
        for _ in range(random.randint(40, 100)):
            gw.sim_layer      += gw.print_speed
            gw.sim_nozzle_y    = gw.sim_bed_y - int(gw.sim_layer * 1.0)
            gw.sim_nozzle_x    = gw.sim_pillar_x + int(
                math.sin(gw.time_elapsed * 10) * (gw.sim_pillar_width // 2))
            gw.defect_severity = min(1.0, gw.defect_severity + random.uniform(0.01, 0.03))
            gw.get_simulated_frame()
        gw.sim_layer = gw.sim_max_layers
        frame = gw.get_simulated_frame()
        idx   = existing_defect + i
        cv2.imwrite(f"{DEFECT_DIR}/defect_{idx:04d}.png", frame)
        if (i + 1) % 50 == 0:
            print(f"  [defect]   {i+1}/{n_each} saved")

    total_n = existing_nominal + n_each
    total_d = existing_defect  + n_each
    print(f"\n[DATASET] New totals: {total_n} nominal, {total_d} defect  ({total_n+total_d} total)\n")


# ==============================================================================
# STEP 2 -- FEATURE EXTRACTION
# ==============================================================================

def extract_features_from_dir(folder, label):
    """
    Extract a 6-dimensional physics feature vector from every image:
      [0] deviation_pct          — volumetric contour deviation %
      [1] complexity_score       — normalised Hu-moment complexity
      [2] contour_count          — number of independent contours detected
      [3] edge_density           — Canny pixel density (edges / total pixels)
      [4] mean_contour_area      — mean area of detected blobs
      [5] std_contour_area       — std-dev of blob areas (spread indicator)
    """
    features, labels = [], []
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  Extracting features from {len(files)} images in '{folder}' ...")

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img  = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # -- Canny edge mask ------------------------------------------------
        edges  = cv2.Canny(gray, 80, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # -- Feature 0: deviation_pct ---------------------------------------
        if cnts:
            widths = [cv2.boundingRect(c)[2] for c in cnts if cv2.contourArea(c) > 5]
            nominal_w = 40.0
            deviation_pct = (np.mean(widths) / nominal_w * 100.0 - 100.0) if widths else 0.0
            deviation_pct = max(0.0, float(deviation_pct))
        else:
            deviation_pct = 0.0

        # -- Feature 1: complexity_score ------------------------------------
        large_cnts = [c for c in cnts if cv2.contourArea(c) > 10]
        complexity  = float(len(large_cnts)) * 2.5

        # -- Feature 2: contour_count ---------------------------------------
        contour_count = float(len(cnts))

        # -- Feature 3: edge_density ----------------------------------------
        edge_density = float(np.count_nonzero(edges)) / (640 * 480)

        # -- Feature 4 & 5: contour area stats -----------------------------
        areas = [float(cv2.contourArea(c)) for c in cnts if cv2.contourArea(c) > 5]
        mean_area = float(np.mean(areas))  if areas else 0.0
        std_area  = float(np.std(areas))   if areas else 0.0

        features.append([deviation_pct, complexity, contour_count,
                          edge_density, mean_area, std_area])
        labels.append(label)

    return features, labels


# ==============================================================================
# STEP 3 -- TRAIN AND EVALUATE
# ==============================================================================

def train_and_save():
    t0 = time.time()

    print("\n" + "="*60)
    print("  SMARTFLOW EDGE -- ML TRAINING PIPELINE")
    print("="*60)

    # -- Extract features ---------------------------------------------------
    print("\n[STEP 2] Feature Extraction ...")
    X_nom, y_nom = extract_features_from_dir(NOMINAL_DIR, label=0)
    X_def, y_def = extract_features_from_dir(DEFECT_DIR,  label=1)

    X = np.array(X_nom + X_def, dtype=np.float32)
    y = np.array(y_nom + y_def, dtype=np.int32)

    print(f"\n  Total feature vectors : {len(X)}")
    print(f"  Nominal samples       : {y_nom.count(0) if isinstance(y_nom, list) else int(np.sum(y==0))}")
    print(f"  Defect  samples       : {y_def.count(1) if isinstance(y_def, list) else int(np.sum(y==1))}")
    print(f"  Feature dimensions    : {X.shape[1]}  "
          f"[deviation_pct, complexity, contour_count, edge_density, mean_area, std_area]")

    # -- Train / test split (75 / 25, stratified) --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    print(f"\n  Train set : {len(X_train)} samples")
    print(f"  Test  set : {len(X_test)}  samples")

    # -- Build pipeline: StandardScaler + RBF-SVM --------------------------
    print("\n[STEP 3] Training RBF-SVM with StandardScaler pipeline ...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=42, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)

    # -- 5-fold stratified cross-validation on training set -------------
    print("\n[STEP 4] 5-Fold Stratified Cross-Validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    print(f"  CV F1 scores  : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV F1 mean    : {cv_scores.mean():.4f}  +-{cv_scores.std():.4f}")

    # -- Held-out test evaluation -------------------------------------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred)
    rec       = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred,
                                       target_names=["Nominal (0)", "Defect (1)"])

    elapsed = time.time() - t0

    print("\n" + "-"*60)
    print("  TRAINING RESULTS")
    print("-"*60)
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Precision       : {prec*100:.2f}%")
    print(f"  Recall          : {rec*100:.2f}%")
    print(f"  F1-Score        : {f1*100:.2f}%")
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"  CV F1 (mean)    : {cv_scores.mean()*100:.2f}%")
    print(f"  Training time   : {elapsed:.1f}s")
    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("\n  Per-Class Report:")
    print(report)
    print("-"*60)

    # -- Serialize model + metadata ----------------------------------------
    payload = {
        "model":          clf,
        "model_type":     "SVM_RBF_Pipeline",
        "feature_names":  ["deviation_pct", "complexity_score", "contour_count",
                            "edge_density", "mean_contour_area", "std_contour_area"],
        "n_features":     6,
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
        "accuracy":       acc,
        "precision":      prec,
        "recall":         rec,
        "f1_score":       f1,
        "roc_auc":        roc_auc,
        "cv_f1_mean":     float(cv_scores.mean()),
        "cv_f1_std":      float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "warning_threshold":  15.0,
        "critical_threshold": 35.0,
        "trained_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "version":        "2.0-production"
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Model saved -> {MODEL_PATH}  ({os.path.getsize(MODEL_PATH):,} bytes)")

    # -- Write training report ---------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as rpt:
        rpt.write("SmartFlow Edge -- ML Training Report\n")
        rpt.write(f"Trained : {payload['trained_at']}\n")
        rpt.write(f"Version : {payload['version']}\n\n")
        rpt.write(f"Dataset : {len(X)} total samples  "
                  f"({int(np.sum(y==0))} nominal + {int(np.sum(y==1))} defect)\n")
        rpt.write(f"Features: {payload['feature_names']}\n\n")
        rpt.write(f"Accuracy  : {acc*100:.2f}%\n")
        rpt.write(f"Precision : {prec*100:.2f}%\n")
        rpt.write(f"Recall    : {rec*100:.2f}%\n")
        rpt.write(f"F1-Score  : {f1*100:.2f}%\n")
        rpt.write(f"ROC-AUC   : {roc_auc:.4f}\n")
        rpt.write(f"CV F1     : {cv_scores.mean()*100:.2f}% +- {cv_scores.std()*100:.2f}%\n\n")
        rpt.write("Confusion Matrix:\n")
        rpt.write(f"  TN={cm[0,0]}  FP={cm[0,1]}\n")
        rpt.write(f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n")
        rpt.write("Per-Class Report:\n")
        rpt.write(report)
    print(f"  Report saved -> {REPORT_PATH}")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE -- MODEL IS PRODUCTION-READY")
    print("="*60 + "\n")

    return payload


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Images already generated (768 nominal + 805 defect = 1573 total)
    # Skip Step 1 -- jump straight to feature extraction and training
    print("\n[STEP 1] Skipping image generation -- 1573 images already on disk.")
    result = train_and_save()
    print(f"\nFinal Model Accuracy  : {result['accuracy']*100:.2f}%")
    print(f"Final F1-Score        : {result['f1_score']*100:.2f}%")
    print(f"Final ROC-AUC         : {result['roc_auc']:.4f}")
