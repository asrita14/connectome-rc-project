import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conn2res.connectivity import Conn # Import Conn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns # Add seaborn
from sklearn.preprocessing import StandardScaler # For scaling reservoir states if needed
from sklearn.decomposition import PCA

# === Setup ===
SEED = 42
np.random.seed(SEED)

# === Get node count from connectome (like mss_random_connected.py) ===
# Set environment variable if needed, assuming 'human' directory is relative
os.environ["CONN2RES_DATA"] = os.path.abspath("human") 
try:
    conn_temp = Conn(subj_id=0)
    N_RESERVOIR = 209 # Determine reservoir size from connectome
    del conn_temp # No longer need the human connectome object
    print(f"Reservoir size determined from connectome: N_RESERVOIR = {N_RESERVOIR}")
except Exception as e:
    print(f"Error loading connectome to determine reservoir size: {e}")
    print("Falling back to default reservoir size N_RESERVOIR = 100")
    N_RESERVOIR = 100 # Fallback size

# === ESN Parameters (rest of them) ===
TARGET_SPECTRAL_RADIUS = 0.95
# N_RESERVOIR = 100 # Define the number of reservoir neurons <-- Removed hardcoded value
INPUT_SCALING = 1.0 # Scale factor for input weights
LEAKING_RATE = 1 # Leaking rate (alpha) for state update

# === Load data ===
# Assuming data is in the same directory or accessible path
try:
    data = np.load("data_2D_manifold_10_classes/1000 samples/spike_data.npy")[:, :, 0]
    labels = np.load("data_2D_manifold_10_classes/1000 samples/spike_labels.npy")
except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data_2D_manifold_10_classes/100 samples/' directory exists and contains the .npy files.")
    exit()

# === Split data ===
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=SEED)
n_inputs = x_train.shape[1]
n_train_samples = x_train.shape[0]
n_test_samples = x_test.shape[0]

print(f"Data loaded: Train samples={n_train_samples}, Test samples={n_test_samples}, Input features={n_inputs}")
print(f"ESN parameters: Reservoir size={N_RESERVOIR}, Spectral radius={TARGET_SPECTRAL_RADIUS}, Leaking rate={LEAKING_RATE}")


# === Generate Random ESN Matrices ===
print("Generating random ESN matrices...")
# Input weights (dense random matrix)
W_in = (np.random.rand(N_RESERVOIR, n_inputs) * 2 - 1) * INPUT_SCALING

# Reservoir weights (sparse random matrix often works well, but using dense here for simplicity)
W = np.random.randn(N_RESERVOIR, N_RESERVOIR)
# Remove self-connections (optional, but common)
# np.fill_diagonal(W, 0)

# Scale spectral radius
eigenvalues = np.linalg.eigvals(W)
current_spectral_radius = np.max(np.abs(eigenvalues))
if current_spectral_radius > 1e-9: # Avoid division by zero
    W = W * (TARGET_SPECTRAL_RADIUS / current_spectral_radius)
else:
    print("Warning: Initial spectral radius is close to zero.")
print(f"Reservoir matrix W generated. Spectral radius scaled to approx {TARGET_SPECTRAL_RADIUS}")

# === Simulate ESN ===
def run_esn(input_data, n_samples):
    """Runs the ESN simulation."""
    states = np.zeros((n_samples, N_RESERVOIR))
    x = np.zeros(N_RESERVOIR) # Initial reservoir state
    for t in range(n_samples):
        u = input_data[t, :]
        # ESN state update equation with leaking rate
        x_pre_activation = W_in @ u + W @ x
        x = (1 - LEAKING_RATE) * x + LEAKING_RATE * np.tanh(x_pre_activation)
        states[t, :] = x
    return states

print("Simulating ESN for training data...")
rs_train = run_esn(x_train, n_train_samples)

print("Simulating ESN for testing data...")
rs_test = run_esn(x_test, n_test_samples)

# Optional: Scale reservoir states (can sometimes improve classifier performance)
# scaler = StandardScaler()
# rs_train = scaler.fit_transform(rs_train)
# rs_test = scaler.transform(rs_test)

# === Train classifier ===
clf = RidgeClassifier(alpha=1.0) # Ridge regression is common for ESN readouts
print("Training classifier (Random ESN)...")
# Use reservoir states as features
clf.fit(rs_train, y_train)
y_pred = clf.predict(rs_test)

# === Evaluation ===
print("\n===== Evaluation Summary (Random ESN) =====")
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
unique_labels = np.unique(np.concatenate((y_train, y_test)))
for label in sorted(unique_labels):
    label_str = str(label)
    if label_str in class_report:
        metrics = class_report[label_str]
        print(f"  Class {label_str}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
print(f"  accuracy: {class_report['accuracy']:.4f}")
print(f"  macro avg:")
for metric_name, value in class_report['macro avg'].items():
    print(f"    {metric_name}: {value:.4f}")
print(f"  weighted avg:")
for metric_name, value in class_report['weighted avg'].items():
    print(f"    {metric_name}: {value:.4f}")


# === Save results ===
output_dir = "figs_random_connected_esn_10_classes"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(output_dir, "predictions_esn.csv"), index=False)

report_df = pd.DataFrame(class_report).transpose()
report_csv_path = os.path.join(output_dir, "classification_report_esn.csv")
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

print(f"Output files will be saved to directory: {output_dir}")


# === Plotting ===

unique_labels_sorted = sorted(np.unique(np.concatenate((y_train, y_test))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels_sorted,
            yticklabels=unique_labels_sorted)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Random ESN)')
cm_save_path = os.path.join(output_dir, "confusion_matrix_esn.png")
plt.savefig(cm_save_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_save_path}")

print("Generating PCA visualization of reservoir states (Random ESN)...")
pca = PCA(n_components=2)
rs_train_2d = pca.fit_transform(rs_train)
rs_test_2d = pca.transform(rs_test)

explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance (Random ESN): PC1 = {explained_var[0]:.2f}, PC2 = {explained_var[1]:.2f}, Total = {sum(explained_var[:2]):.2f}")

plt.figure(figsize=(12, 10))
unique_labels_train = np.unique(y_train)
unique_labels_test = np.unique(y_test)

for label in unique_labels_train:
    indices = np.where(y_train == label)[0]
    plt.scatter(rs_train_2d[indices, 0], rs_train_2d[indices, 1],
               label=f'Train Class {label}', alpha=0.7, s=50)

for label in unique_labels_test:
    indices = np.where(y_test == label)[0]
    plt.scatter(rs_test_2d[indices, 0], rs_test_2d[indices, 1],
               label=f'Test Class {label}', alpha=0.3, s=25, marker='x')

plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
plt.title('PCA Projection of Reservoir States (Random ESN)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 0.85, 1])
pca_save_path = os.path.join(output_dir, "reservoir_states_pca_esn.png")
plt.savefig(pca_save_path)
plt.close()
print(f"PCA projection saved to {pca_save_path}")

print("Script finished.")
