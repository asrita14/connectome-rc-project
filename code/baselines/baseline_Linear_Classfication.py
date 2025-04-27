import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.preprocessing import StandardScaler # Optional: For scaling input features

# === Setup ===
SEED = 42
np.random.seed(SEED)

# === Output Directory ===
OUTPUT_DIR = "figs_baseline_linear"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
# Assume data is in 'data_2D_manifold/600 samples/' relative to script location
DATA_DIR = "data_1D_manifold_2_classes_more_samples" # Adjust if necessary
try:
    # Assuming spike_data.npy is (samples, features) - check if needs flattening
    data = np.load(os.path.join(DATA_DIR, "spike_data.npy"))
    labels = np.load(os.path.join(DATA_DIR, "spike_labels.npy"))
    print(f"Original data shape: {data.shape}")

    # Check if data needs flattening (e.g., if it's time series per sample)
    if data.ndim > 2:
        # Example: Flatten last two dimensions if shape is (samples, time_steps, features)
        data = data.reshape(data.shape[0], -1)
        print(f"Warning: Data has {data.ndim} dimensions. Flattening to {data.shape}.")
        # Adjust flattening strategy based on your data's actual structure
        # For now, assuming it might be (samples, neurons) already suitable
        # if data.shape[1] != 30: # Based on your previous context of 30 neurons
        #      print(f"Warning: Number of features ({data.shape[1]}) doesn't match expected 30. Adjust loading/flattening if needed.")

except FileNotFoundError:
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

# === Optional: Scale input features ===
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print("Input features scaled.")

# === Train Classifier ===
# Using RidgeClassifier as per the old draft
clf = RidgeClassifier(alpha=1.0)
print("Training Linear Classifier (Ridge)...")
clf.fit(x_train, y_train)

# === Predict ===
y_pred = clf.predict(x_test)

# === Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== Evaluation Summary (Linear Classifier) =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# === Save Results ===
results_summary = {
    "model": "Linear Classifier (Ridge)",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "micro_f1": micro_f1,
    "weighted_f1": weighted_f1,
    "classification_report": report,
    "confusion_matrix": cm.tolist() # Convert numpy array to list for saving
}
# Save metrics to a text file
metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics_linear.txt")
with open(metrics_path, "w") as f:
    f.write("===== Evaluation Summary (Linear Classifier) =====\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Micro F1: {micro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
print(f"Evaluation metrics saved to {metrics_path}")

# Save predictions
predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
predictions_path = os.path.join(OUTPUT_DIR, "predictions_linear.csv")
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")

# === Plot Confusion Matrix ===
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
fig, ax = plt.subplots(figsize=(6, 5))
display.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Linear Classifier (Ridge)")
plt.tight_layout()
cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix_linear.png")
plt.savefig(cm_plot_path)
print(f"Confusion matrix plot saved to {cm_plot_path}")
# plt.show() # Optionally display the plot

print("\nBaseline Linear Classification script finished.")
