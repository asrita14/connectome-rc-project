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
data = np.load("data_2D_manifold_10_classes/1000 samples/spike_data.npy")[:, :, 0]
labels = np.load("data_2D_manifold_10_classes/1000 samples/spike_labels.npy")

# === Split data ===
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=SEED)

# === Load and normalize connectivity ===
conn = Conn(subj_id=0)
conn.scale_and_normalize()

# === Use visual and default mode networks as input nodes ===
vis_nodes = conn.get_nodes('VIS')
dmn_nodes = conn.get_nodes('DMN')
available_nodes = np.union1d(vis_nodes, dmn_nodes)

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
clf = RidgeClassifier(alpha=1.0)
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
output_dir = "figs_vis_dmn_10_classes"
os.makedirs(output_dir, exist_ok=True)
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

report_df = pd.DataFrame(class_report).transpose()
report_csv_path = os.path.join(output_dir, "classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

print(f"Output files will be saved to directory: {output_dir}")

# === Plotting ===
plotting.plot_reservoir_states(
    x=x_train, reservoir_states=rs_train,
    title="Training Reservoir States (VIS+DMN, 10 Classes)",
    savefig=True,
    fname=os.path.join(output_dir, "res_states_train.png"),
    show=False
)
plotting.plot_reservoir_states(
    x=x_test, reservoir_states=rs_test,
    title="Testing Reservoir States (VIS+DMN, 10 Classes)",
    savefig=True,
    fname=os.path.join(output_dir, "res_states_test.png"),
    show=False
)

# Plot Confusion Matrix
unique_labels_sorted = sorted(np.unique(np.concatenate((y_train, y_test))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels_sorted,
            yticklabels=unique_labels_sorted)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (MSS VIS+DMN, 10 Classes)')
cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_save_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_save_path}")

# === PCA visualization of reservoir states ===
print("Generating PCA visualization of reservoir states...")
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
plt.title('PCA Projection of Reservoir States (MSS VIS+DMN, 10 Classes)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 0.85, 1])
pca_save_path = os.path.join(output_dir, "reservoir_states_pca.png")
plt.savefig(pca_save_path)
plt.close()
print(f"PCA projection saved to {pca_save_path}")
