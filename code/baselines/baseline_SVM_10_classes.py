import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.preprocessing import StandardScaler # Optional: For scaling input features

# === Setup ===
SEED = 42
np.random.seed(SEED)

# === Output Directory ===
OUTPUT_DIR = "figs_baseline_svm_10cls" # Updated output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
# Updated data directory
DATA_DIR = "data_2D_manifold_10_classes/1000 samples"
try:
    # Load data, expecting (samples, features, 2)
    data = np.load(os.path.join(DATA_DIR, "spike_data.npy"))
    labels = np.load(os.path.join(DATA_DIR, "spike_labels.npy"))
    print(f"Original data shape: {data.shape}")

    # Check if data needs flattening
    if data.ndim > 2:
        # Flatten dimensions beyond the first (samples) dimension
        original_shape = data.shape
        data = data.reshape(data.shape[0], -1)
        print(f"Data flattened from {original_shape} to {data.shape}.")
    elif data.ndim < 2:
         raise ValueError(f"Loaded data has {data.ndim} dimensions, expected at least 2.")

except FileNotFoundError:
    # Updated error message
    print(f"Error: Data files not found in {DATA_DIR}. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# === Split data ===
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=SEED, stratify=labels)
n_train_samples, n_features = x_train.shape
n_test_samples = x_test.shape[0]

print(f"Data loaded: Train samples={n_train_samples}, Test samples={n_test_samples}, Input features={n_features}")

# === Optional: Scale input features (Often recommended for SVM) ===
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Input features scaled.")

# === Train Classifier ===
# Using SVC with default RBF kernel as per the old draft
clf = SVC(random_state=SEED)
print("Training SVM Classifier...")
clf.fit(x_train, y_train)

# === Predict ===
y_pred = clf.predict(x_test)

# === Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
# Add zero_division=0
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
# Get report as dict and string
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_str = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== Evaluation Summary (SVM Classifier, 10 Classes) =====") # Updated title
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report_str)

# === Save Results ===
# Save classification report to CSV
report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = os.path.join(OUTPUT_DIR, "classification_report_svm_10cls.csv") # Updated filename
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

# Save predictions
predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
predictions_path = os.path.join(OUTPUT_DIR, "predictions_svm_10cls.csv") # Updated filename
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")

# === Plot Confusion Matrix ===
unique_labels_sorted = sorted(np.unique(labels))
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels_sorted)
fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size for 10 classes
display.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM Classifier (RBF, 10 Classes)") # Updated title
plt.tight_layout()
cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix_svm_10cls.png") # Updated filename
plt.savefig(cm_plot_path)
plt.close() # Close figure
print(f"Confusion matrix plot saved to {cm_plot_path}")

print("\nBaseline SVM script (10 classes) finished.")
