import os
import cv2
import numpy as np
import pickle
from src.vision.detector import DefectDetector

# Ensure scikit-learn is installed. If not, fallback to manual boundary classifier.
try:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class EdgeMLTrainer:
    """
    Edge-native Machine Learning trainer.
    Extracts high-quality mathematical features from 3D printing images
    and trains a lightweight Support Vector Machine (SVM) classifier
    optimized to run on low-resource (8 GB RAM) hardware.
    """
    def __init__(self):
        self.detector = DefectDetector()
        self.features = []
        self.labels = []

    def extract_features_from_folder(self, folder_path, label):
        """
        Processes a directory of images, running the CV pipeline to extract
        low-dimensional mathematical print state telemetry.
        label: 0 for healthy/nominal, 1 for spaghetti defect.
        """
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}. Skipping.")
            return
            
        print(f"Extracting edge features from: {folder_path}...")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        count = 0
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            # Resize to pipeline workspace size
            frame = cv2.resize(frame, (640, 480))
            
            # Process with our OpenCV spectrometer
            _, telemetry = self.detector.analyze_frame(frame)
            
            # Extract the 4-dimensional physics feature vector
            feat_vector = [
                telemetry["deviation_pct"],
                telemetry["complexity_score"],
                telemetry["contour_count"],
                telemetry["average_deviation_5s"]
            ]
            
            self.features.append(feat_vector)
            self.labels.append(label)
            count += 1
            
        print(f"Successfully processed {count} images.")

    def train_and_save(self, dataset_dir="data/dataset"):
        """
        Runs the feature extraction, trains the SVM classifier, and serializes the model.
        """
        nominal_dir = os.path.join(dataset_dir, "nominal")
        defect_dir = os.path.join(dataset_dir, "defect")
        
        # 1. Feature Extraction
        self.extract_features_from_folder(nominal_dir, label=0)
        self.extract_features_from_folder(defect_dir, label=1)
        
        if len(self.features) == 0:
            print("Error: No training images found. Please populate 'data/dataset/nominal' and 'data/dataset/defect' folders first.")
            return False
            
        X = np.array(self.features)
        y = np.array(self.labels)
        
        model_path = os.path.join("data", "defect_classifier.pkl")
        
        if SKLEARN_AVAILABLE:
            print("Scikit-Learn detected. Training robust Support Vector Machine (SVM) edge classifier...")
            
            # Split for internal validation
            if len(X) > 4:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            else:
                X_train, y_train = X, y
                X_test, y_test = X, y
                
            # Train the SVM model
            clf = SVC(probability=True, kernel='rbf', C=1.0)
            clf.fit(X_train, y_train)
            
            # Output training accuracy
            score = clf.score(X_test, y_test)
            print(f"Edge Classifier trained successfully. Validation Accuracy: {score * 100.0:.2f}%")
            
            # Save the trained model binary
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)
            print(f"Model serialized and saved to: {model_path}")
            return True
        else:
            print("Scikit-Learn not found. Creating a rule-based decision threshold model...")
            # If sklearn is missing, we create a deterministic decision tree threshold wrapper
            # to preserve compatibility
            mock_model = {
                "type": "heuristic",
                "warning_limit": 15.0,
                "critical_limit": 35.0
            }
            with open(model_path, 'wb') as f:
                pickle.dump(mock_model, f)
            print(f"Heuristic decision model saved to: {model_path}")
            return True

if __name__ == "__main__":
    # Setup folders automatically
    os.makedirs("data/dataset/nominal", exist_ok=True)
    os.makedirs("data/dataset/defect", exist_ok=True)
    
    print("--- Edge ML Training Utility ---")
    print("Please place your healthy print frames inside: 'data/dataset/nominal/'")
    print("Please place your spaghetti fail frames inside: 'data/dataset/defect/'")
    print("---------------------------------")
    
    trainer = EdgeMLTrainer()
    trainer.train_and_save()
