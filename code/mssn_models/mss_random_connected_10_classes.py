import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conn2res.connectivity import Conn  # Keep Conn temporarily to get node count
from conn2res.reservoir import MSSNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
# from scipy.linalg import eigvals # Using numpy instead

# === Setup ===
os.environ["CONN2RES_DATA"] = os.path.abspath("human")
SEED = 42
np.random.seed(SEED)
TARGET_SPECTRAL_RADIUS = 0.95

# === Load data ===
data = np.load("data_2D_manifold_10_classes/1000 samples/spike_data.npy")[:, :, 0]
labels = np.load("data_2D_manifold_10_classes/1000 samples/spike_labels.npy")

# === Split data ===
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=SEED)

# === Get node count from original connectome ===
conn_temp = Conn(subj_id=0)
n_nodes = 427
del conn_temp # No longer need the human connectome object

# === Generate Random Connectivity Matrix ===
print(f"Generating random connectivity matrix with {n_nodes} nodes...")
w_random = np.random.randn(n_nodes, n_nodes)
np.fill_diagonal(w_random, 0) # No self-connections

# Scale spectral radius
eigenvalues = np.linalg.eigvals(w_random)
current_spectral_radius = np.max(np.abs(eigenvalues))
if current_spectral_radius > 1e-9: # Avoid division by zero
    w_random = w_random * (TARGET_SPECTRAL_RADIUS / current_spectral_radius)
print(f"Random matrix generated. Spectral radius scaled to approx {TARGET_SPECTRAL_RADIUS}")


# === Randomly select nodes ===
all_node_indices = np.arange(n_nodes)
n_requested = min(x_train.shape[1], n_nodes - 1) # Ensure at least 1 node left for gr/int

if n_requested <= 0:
    raise ValueError("Input data dimension is zero or negative.")

# Select random external nodes
ext_nodes = np.random.choice(all_node_indices, n_requested, replace=False)

# Select a random grounding node from remaining nodes
available_for_gr = np.setdiff1d(all_node_indices, ext_nodes)
if len(available_for_gr) == 0:
     raise ValueError(f"Cannot select grounding node, only {n_nodes} total nodes and requested {n_requested} external nodes.")
gr_nodes = np.random.choice(available_for_gr, 1, replace=False)

# Select internal nodes (all remaining nodes)
int_nodes = np.setdiff1d(all_node_indices, np.union1d(ext_nodes, gr_nodes))

print(f"Selected nodes: {len(ext_nodes)} external, {len(gr_nodes)} grounding, {len(int_nodes)} internal.")

# === Adjust inputs ===
# Ensure input data matches the number of external nodes
x_train = x_train[:, :n_requested]
x_test = x_test[:, :n_requested]

# === MSSNetwork ===
mssn = MSSNetwork(
    w=w_random, # Use the random matrix
    int_nodes=int_nodes,
    ext_nodes=ext_nodes,
    gr_nodes=gr_nodes,
    mode='forward'
)

# === Simulate ===
print("Simulating training (Randomly Connected Network)...")
rs_train = mssn.simulate(Vext=x_train)
print("Simulating testing (Randomly Connected Network)...")
rs_test = mssn.simulate(Vext=x_test)

# === Train classifier ===
clf = RidgeClassifier(alpha=1.0)
print("Training classifier (Randomly Connected Network)...")
clf.fit(rs_train, y_train)
y_pred = clf.predict(rs_test)

# === Evaluation ===
print("\n===== Evaluation Summary (Randomly Connected MSS) =====")
accuracy = accuracy_score(y_test, y_pred)
# Calculate additional F1 scores and confusion matrix
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
# Get report as dict
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Print summary metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# Print detailed classification report from dict
print("\nClassification Report:")
unique_labels_print = np.unique(np.concatenate((y_train, y_test)))
for label in sorted(unique_labels_print):
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
output_dir = "figs_random_connected_mss_10cls"
os.makedirs(output_dir, exist_ok=True)
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# Save classification report to CSV
report_df = pd.DataFrame(class_report).transpose()
report_csv_path = os.path.join(output_dir, "classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

print(f"Output files will be saved to directory: {output_dir}")

# === Plotting ===
# Optional: Keep or remove these original plotting functions
# plotting.plot_reservoir_states(...)

# Plot Confusion Matrix
unique_labels_sorted = sorted(np.unique(np.concatenate((y_train, y_test))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels_sorted,
            yticklabels=unique_labels_sorted)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Randomly Connected MSS, 10 Classes)')
cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_save_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_save_path}")


# === PCA visualization of reservoir states ===
print("Generating PCA visualization of reservoir states (Randomly Connected Network)...")
# 使用PCA降维到2维用于可视化
pca = PCA(n_components=2)
rs_train_2d = pca.fit_transform(rs_train)
rs_test_2d = pca.transform(rs_test)

# 计算解释方差比例
explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance (Randomly Connected): PC1 = {explained_var[0]:.2f}, PC2 = {explained_var[1]:.2f}, Total = {sum(explained_var[:2]):.2f}")

# 绘制PCA降维后的储备层状态
plt.figure(figsize=(12, 10))

# 获取唯一的类别标签
unique_labels = np.unique(y_train)

# 绘制训练集
for label in unique_labels:
    indices = np.where(y_train == label)[0]
    plt.scatter(rs_train_2d[indices, 0], rs_train_2d[indices, 1],
               label=f'Train Class {label}', alpha=0.7, s=50)

# 绘制测试集
unique_test_labels = np.unique(y_test)
for label in unique_test_labels:
    indices = np.where(y_test == label)[0]
    plt.scatter(rs_test_2d[indices, 0], rs_test_2d[indices, 1],
               label=f'Test Class {label}', alpha=0.3, s=25, marker='x')

plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
plt.title('PCA Projection of Reservoir States (Randomly Connected Network, 10 Classes)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 0.85, 1])
pca_save_path = os.path.join(output_dir, "reservoir_states_pca.png")
plt.savefig(pca_save_path)
plt.close()
print(f"PCA projection saved to {pca_save_path}")
