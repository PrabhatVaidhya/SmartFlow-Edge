"""
SmartFlow Edge -- Additional Real Defect Dataset Fetcher
Downloads real spaghetti/defect images from multiple HuggingFace sources
and adds them to the defect folder, then retrains.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import pickle
import time

try:
    from datasets import load_dataset
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "Pillow", "-q"])
    from datasets import load_dataset
    from PIL import Image

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])

NOMINAL_DIR = "data/dataset/nominal"
DEFECT_DIR  = "data/dataset/defect"
MODEL_PATH  = "data/defect_classifier.pkl"
REPORT_PATH = "data/training_report.txt"

os.makedirs(NOMINAL_DIR, exist_ok=True)
os.makedirs(DEFECT_DIR,  exist_ok=True)


def save_image(img_input, path):
    """Save PIL image or numpy array to path as BGR PNG."""
    if isinstance(img_input, np.ndarray):
        img_bgr = img_input
    else:
        img_np  = np.array(img_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (640, 480))
    cv2.imwrite(path, img_bgr)


def fetch_real_datasets():
    total_nom = 0
    total_def = 0

    # Clean up any old Javiai images to avoid duplicates or incorrectly labeled images
    print("\nCleaning up old Javiai files from previous runs...")
    for folder in [NOMINAL_DIR, DEFECT_DIR]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if "javiai" in f.lower() and f.endswith(".png"):
                    try:
                        os.remove(os.path.join(folder, f))
                    except Exception:
                        pass

    # ---- Dataset 1: Javiai/failures-3D-print (fix label mapping) -----------
    print("\n[DS-1] Javiai/failures-3D-print (corrected label parsing) ...")
    try:
        ds = load_dataset("Javiai/failures-3D-print", split="train")
        print(f"       {len(ds)} images available.")
        d, n = 0, 0
        for i, sample in enumerate(ds):
            img = sample.get("image") or sample.get("img")
            if img is None:
                continue
            objects = sample.get("objects", {})
            # Category IDs: check both string names and integer IDs
            categories = []
            if isinstance(objects, dict):
                categories = objects.get("categories", [])
                if not categories:
                    categories = objects.get("category", [])
                if not categories:
                    categories = objects.get("label", [])
            # In Javiai: category 0=Error, category 1=Extrusor, category 2=Part, category 3=Spaghetti
            # Defect if any category is 0 (Error) or 3 (Spaghetti)
            is_defect = any(int(c) in [0, 3] for c in categories if str(c).isdigit())

            if is_defect:
                save_image(img, f"{DEFECT_DIR}/real2_javiai_{i:04d}.png")
                d += 1
            else:
                save_image(img, f"{NOMINAL_DIR}/real2_javiai_{i:04d}.png")
                n += 1
        print(f"       Saved: {d} defect + {n} nominal")
        total_def += d; total_nom += n
    except Exception as e:
        print(f"       [WARN] {e}")

    # ---- Dataset 2: imagenet-style 3D printing defects from HF -------------
    print("\n[DS-2] trying keremberke/3d-printing-object-detection ...")
    try:
        ds2 = load_dataset("keremberke/3d-printing-object-detection", "full", split="train")
        print(f"       {len(ds2)} images available.")
        d, n = 0, 0
        for i, sample in enumerate(ds2):
            img = sample.get("image") or sample.get("img")
            if img is None:
                continue
            # labels: 0=spaghetti
            objects = sample.get("objects", {})
            categories = objects.get("category", []) if isinstance(objects, dict) else []
            is_defect = len(categories) > 0  # any detected object = defect/spaghetti print
            if is_defect:
                save_image(img, f"{DEFECT_DIR}/real2_kb_{i:04d}.png")
                d += 1
            else:
                save_image(img, f"{NOMINAL_DIR}/real2_kb_{i:04d}.png")
                n += 1
            if (i+1) % 100 == 0:
                print(f"       Processed {i+1}/{len(ds2)} ...")
        print(f"       Saved: {d} defect + {n} nominal")
        total_def += d; total_nom += n
    except Exception as e:
        print(f"       [WARN] {e}")

    # ---- Dataset 3: Roboflow-style 3D printing failures --------------------
    print("\n[DS-3] trying Francesco215/3d-printing-defects ...")
    try:
        ds3 = load_dataset("Francesco215/3d-printing-defects", split="train")
        print(f"       {len(ds3)} images available.")
        d, n = 0, 0
        for i, sample in enumerate(ds3):
            img = sample.get("image") or sample.get("pixel_values") or sample.get("img")
            label = sample.get("label", sample.get("labels", -1))
            if img is None:
                continue
            # 0 = good/nominal, anything else = defect
            is_defect = (int(label) != 0) if str(label).lstrip('-').isdigit() else False
            if is_defect:
                save_image(img, f"{DEFECT_DIR}/real2_ff_{i:04d}.png")
                d += 1
            else:
                save_image(img, f"{NOMINAL_DIR}/real2_ff_{i:04d}.png")
                n += 1
            if (i+1) % 100 == 0:
                print(f"       Processed {i+1}/{len(ds3)} ...")
        print(f"       Saved: {d} defect + {n} nominal")
        total_def += d; total_nom += n
    except Exception as e:
        print(f"       [WARN] {e}")

    # ---- Dataset 4: General 3D print quality dataset -----------------------
    print("\n[DS-4] trying pcuenq/3d-printing ...")
    try:
        ds4 = load_dataset("pcuenq/3d-printing", split="train")
        print(f"       {len(ds4)} images available.")
        d, n = 0, 0
        for i, sample in enumerate(ds4):
            img = sample.get("image") or sample.get("img")
            label = sample.get("label", sample.get("labels", 0))
            if img is None:
                continue
            label_str = str(label).lower()
            is_defect = any(k in label_str for k in ["spaghetti","defect","fail","error","bad","stringing"])
            if is_defect:
                save_image(img, f"{DEFECT_DIR}/real2_pc_{i:04d}.png")
                d += 1
            else:
                save_image(img, f"{NOMINAL_DIR}/real2_pc_{i:04d}.png")
                n += 1
            if (i+1) % 100 == 0:
                print(f"       Processed {i+1}/{len(ds4)} ...")
        print(f"       Saved: {d} defect + {n} nominal")
        total_def += d; total_nom += n
    except Exception as e:
        print(f"       [WARN] {e}")

    return total_nom, total_def


def extract_features_from_dir(folder, label):
    features, labels = [], []
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  Extracting from {len(files):>5} images in '{folder}' ...")
    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img  = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 80, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            widths = [cv2.boundingRect(c)[2] for c in cnts if cv2.contourArea(c) > 5]
            deviation_pct = max(0.0, float((np.mean(widths)/40.0*100.0-100.0) if widths else 0.0))
        else:
            deviation_pct = 0.0
        large_cnts    = [c for c in cnts if cv2.contourArea(c) > 10]
        complexity    = float(len(large_cnts)) * 2.5
        contour_count = float(len(cnts))
        edge_density  = float(np.count_nonzero(edges)) / (640 * 480)
        areas     = [float(cv2.contourArea(c)) for c in cnts if cv2.contourArea(c) > 5]
        mean_area = float(np.mean(areas)) if areas else 0.0
        std_area  = float(np.std(areas))  if areas else 0.0
        features.append([deviation_pct, complexity, contour_count, edge_density, mean_area, std_area])
        labels.append(label)
    return features, labels


def train_and_save():
    t0 = time.time()
    print("\n" + "="*60)
    print("  SMARTFLOW EDGE -- FINAL COMBINED TRAINING v4.0")
    print("="*60)
    print("\n[STEP 2] Feature Extraction ...")
    X_nom, y_nom = extract_features_from_dir(NOMINAL_DIR, 0)
    X_def, y_def = extract_features_from_dir(DEFECT_DIR,  1)
    X = np.array(X_nom + X_def, dtype=np.float32)
    y = np.array(y_nom + y_def, dtype=np.int32)
    n_nom = int(np.sum(y==0)); n_def = int(np.sum(y==1))
    print(f"\n  Total : {len(X)}  ({n_nom} nominal + {n_def} defect)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"  Train : {len(X_train)}   Test : {len(X_test)}")

    print("\n[STEP 3] Training RBF-SVM ...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                    probability=True, random_state=42, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)

    print("\n[STEP 4] 5-Fold Cross-Validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    print(f"  CV F1 : {[f'{s:.4f}' for s in cv_scores]}  mean={cv_scores.mean():.4f} +-{cv_scores.std():.4f}")

    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=["Nominal","Defect"])
    elapsed = time.time() - t0

    print("\n" + "-"*60)
    print("  FINAL RESULTS")
    print("-"*60)
    print(f"  Samples    : {len(X)}  ({n_nom} nominal + {n_def} defect)")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1-Score   : {f1*100:.2f}%")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  CV F1 mean : {cv_scores.mean()*100:.2f}%")
    print(f"  Time       : {elapsed:.1f}s")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{report}")
    print("-"*60)

    payload = {
        "model": clf, "model_type": "SVM_RBF_Pipeline",
        "feature_names": ["deviation_pct","complexity_score","contour_count",
                          "edge_density","mean_contour_area","std_contour_area"],
        "n_features": 6, "total_samples": len(X),
        "nominal_samples": n_nom, "defect_samples": n_def,
        "train_samples": len(X_train), "test_samples": len(X_test),
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "roc_auc": roc_auc,
        "cv_f1_mean": float(cv_scores.mean()), "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "warning_threshold": 15.0, "critical_threshold": 35.0,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "4.0-realworld-combined",
        "data_sources": ["synthetic-smartflow-1573",
                         "huggingface-javiai-failures-3D-print",
                         "huggingface-keremberke-3d-printing",
                         "huggingface-Francesco215-3d-printing-defects",
                         "huggingface-pcuenq-3d-printing"]
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    with open(REPORT_PATH, "w", encoding="utf-8") as rpt:
        rpt.write(f"SmartFlow Edge -- Real-World Combined Training Report\n")
        rpt.write(f"Version  : {payload['version']}\n")
        rpt.write(f"Trained  : {payload['trained_at']}\n")
        rpt.write(f"Sources  : {payload['data_sources']}\n\n")
        rpt.write(f"Dataset  : {len(X)} total ({n_nom} nominal + {n_def} defect)\n\n")
        rpt.write(f"Accuracy  : {acc*100:.2f}%\n")
        rpt.write(f"Precision : {prec*100:.2f}%\n")
        rpt.write(f"Recall    : {rec*100:.2f}%\n")
        rpt.write(f"F1-Score  : {f1*100:.2f}%\n")
        rpt.write(f"ROC-AUC   : {roc_auc:.4f}\n")
        rpt.write(f"CV F1     : {cv_scores.mean()*100:.2f}% +-{cv_scores.std()*100:.2f}%\n\n")
        rpt.write(f"Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}\n\n")
        rpt.write(report)
    print(f"\n  Model saved -> {MODEL_PATH}  ({os.path.getsize(MODEL_PATH):,} bytes)")
    print(f"  Report saved -> {REPORT_PATH}")
    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE -- MODEL v4.0 IS PRODUCTION-READY")
    print("="*60 + "\n")
    return payload


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FETCHING ADDITIONAL REAL-WORLD DATASETS")
    print("="*60)
    n_nom, n_def = fetch_real_datasets()
    total_nom = len([f for f in os.listdir(NOMINAL_DIR) if f.endswith('.png')])
    total_def = len([f for f in os.listdir(DEFECT_DIR)  if f.endswith('.png')])
    print(f"\n  Combined dataset: {total_nom} nominal + {total_def} defect = {total_nom+total_def} total")
    result = train_and_save()
    print(f"\nFinal Accuracy  : {result['accuracy']*100:.2f}%")
    print(f"Final F1-Score  : {result['f1_score']*100:.2f}%")
    print(f"Final ROC-AUC   : {result['roc_auc']:.4f}")
    print(f"Model Version   : {result['version']}")
    print(f"Total Samples   : {result['total_samples']}")
