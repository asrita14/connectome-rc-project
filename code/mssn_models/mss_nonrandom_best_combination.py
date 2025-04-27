import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conn2res.connectivity import Conn
from conn2res.reservoir import MSSNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA

# === Setup ===
os.environ["CONN2RES_DATA"] = os.path.abspath("human")
SEED = 42
np.random.seed(SEED)

# === Load data ===
try:
    data = np.load("data_1D_manifold_2_classes_more_samples/spike_data.npy")[:, :, 0]
    labels = np.load("data_1D_manifold_2_classes_more_samples/spike_labels.npy")
except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data_1D_manifold_2_classes_more_samples' directory exists and contains the .npy files.")
    exit()

# === Split data ===
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=SEED)

# === Load and normalize connectivity ===
conn = Conn(subj_id=0)
conn.scale_and_normalize()

# === Use visual and default mode networks as input nodes ===
vis_nodes = conn.get_nodes('VIS')
dmn_nodes = conn.get_nodes('DMN')
available_nodes = np.union1d(vis_nodes, dmn_nodes)
print(f"Total unique nodes in VIS + DMN: {len(available_nodes)}")

n_requested = min(x_train.shape[1], len(available_nodes))
ext_nodes = conn.get_nodes('random', nodes_from=available_nodes, n_nodes=n_requested)
gr_nodes = conn.get_nodes('random', nodes_from=conn.get_nodes('ctx'), n_nodes=1)
int_nodes = conn.get_nodes('all', nodes_without=np.union1d(ext_nodes, gr_nodes), n_nodes=n_requested)

# === Adjust inputs ===
x_train = x_train[:, :n_requested]
x_test = x_test[:, :n_requested]

# === MSSNetwork ===
mssn = MSSNetwork(
    w=conn.w,
    int_nodes=int_nodes,
    ext_nodes=ext_nodes,
    gr_nodes=gr_nodes,
    mode='forward'
)

# === Simulate ===
print("Simulating training...")
rs_train = mssn.simulate(Vext=x_train)
print("Simulating testing...")
rs_test = mssn.simulate(Vext=x_test)

# === Train classifier ===
clf = RidgeClassifier(alpha=1.0, random_state=SEED)
print("Training classifier...")
clf.fit(rs_train, y_train)
y_pred = clf.predict(rs_test)

# === Evaluation ===
print("\n===== Evaluation Summary (MSS VIS+DMN) =====")
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
report_df_display = pd.DataFrame(class_report).transpose()
print(report_df_display.to_string(float_format="{:.4f}".format))

# === Save results ===
output_dir = "figs_mss_vis_dmn"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(output_dir, "predictions_mss.csv"), index=False)

report_df = pd.DataFrame(class_report).transpose()
report_csv_path = os.path.join(output_dir, "classification_report_mss.csv")
report_df.to_csv(report_csv_path)
print(f"\nClassification report saved to {report_csv_path}")

print(f"Output files will be saved to directory: {output_dir}")

# === Plotting ===

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (MSS VIS+DMN)')
cm_save_path = os.path.join(output_dir, "confusion_matrix_mss.png")
plt.savefig(cm_save_path)
print(f"Confusion matrix plot saved to {cm_save_path}")
plt.close()

print("\nGenerating PCA visualization of reservoir states...")
pca = PCA(n_components=2)
rs_train_2d = pca.fit_transform(rs_train)
rs_test_2d = pca.transform(rs_test)

explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1 = {explained_var[0]:.2f}, PC2 = {explained_var[1]:.2f}, Total = {sum(explained_var[:2]):.2f}")

plt.figure(figsize=(12, 10))

unique_labels = np.unique(y_train)

for label in unique_labels:
    indices = np.where(y_train == label)[0]
    plt.scatter(rs_train_2d[indices, 0], rs_train_2d[indices, 1], 
               label=f'Train Class {label}', alpha=0.7, s=50)

unique_test_labels = np.unique(y_test)
for label in unique_test_labels:
    indices = np.where(y_test == label)[0]
    plt.scatter(rs_test_2d[indices, 0], rs_test_2d[indices, 1], 
               label=f'Test Class {label}', alpha=0.3, s=25, marker='x')

plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
plt.title('PCA Projection of Reservoir States (MSS VIS+DMN)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
pca_save_path = os.path.join(output_dir, "reservoir_states_pca_mss.png")
plt.savefig(pca_save_path)
print(f"PCA projection saved to {pca_save_path}")
plt.close()

print("\nScript finished.")
