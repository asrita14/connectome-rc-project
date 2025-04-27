import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conn2res.connectivity import Conn # To determine reservoir size
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm # For progress bar during simulation
import math

# === Setup ===
SEED = 42
np.random.seed(SEED)
TARGET_SPECTRAL_RADIUS = 0.95
os.environ["CONN2RES_DATA"] = os.path.abspath("human")
DT = 1.0 # Simulation time step (ms)

# === Get node count from connectome ===
try:
    conn_temp = Conn(subj_id=0)
    N_NODES = 276 # Number of neurons in the reservoir
    del conn_temp
    print(f"Reservoir size determined from connectome: N_NODES = {N_NODES}")
except Exception as e:
    print(f"Error loading connectome to determine reservoir size: {e}")
    print("Falling back to default reservoir size N_NODES = 100")
    N_NODES = 100 # Fallback size

# === Generate Random Connectivity Matrix ===
print(f"Generating random connectivity matrix W_rec with {N_NODES} nodes...")
W_rec = np.random.randn(N_NODES, N_NODES)
# Scale weights slightly (can be adjusted)
W_rec *= 0.8 / np.sqrt(N_NODES) # Common scaling heuristic
np.fill_diagonal(W_rec, 0) # No self-connections

# Scale spectral radius
try:
    eigenvalues = np.linalg.eigvals(W_rec)
    current_spectral_radius = np.max(np.abs(eigenvalues))
    if current_spectral_radius > 1e-9: # Avoid division by zero
        W_rec = W_rec * (TARGET_SPECTRAL_RADIUS / current_spectral_radius)
    print(f"Recurrent matrix W_rec generated. Spectral radius scaled to approx {TARGET_SPECTRAL_RADIUS}")
except np.linalg.LinAlgError:
     print("Warning: Eigenvalue computation failed for W_rec. Using unscaled matrix.")

# === Load data ===
print("Loading spike event data...")
try:
    # data shape expected: (n_samples, n_features, 2) where [:,:,0] is time, [:,:,1] is neuron ID
    raw_data = np.load("data_2D_manifold_10_classes/1000 samples/spike_data.npy")
    labels = np.load("data_2D_manifold_10_classes/1000 samples/spike_labels.npy")

    if raw_data.ndim != 3 or raw_data.shape[2] != 2:
        raise ValueError(f"Loaded data has unexpected shape: {raw_data.shape}. Expected (samples, features, 2).")

    n_samples, n_input_features, _ = raw_data.shape
    print(f"Loaded raw data shape: {raw_data.shape} (samples, features, [time, id])")
    print(f"Number of input features: {n_input_features}")
    print(f"Loaded labels shape: {labels.shape}")

    # Determine simulation duration based on max spike time
    max_spike_time = np.max(raw_data[:, :, 0])
    SIM_DURATION_STEPS = math.ceil(max_spike_time / DT) + 10 # Add a small buffer
    print(f"Max spike time found: {max_spike_time:.2f} ms. Simulation duration: {SIM_DURATION_STEPS} steps ({SIM_DURATION_STEPS * DT} ms)")

    # Preprocess spike data for efficient lookup
    print("Preprocessing spike data...")
    input_spike_trains = [[] for _ in range(n_samples)]
    for i in range(n_samples):
        sample_spikes = {} # {time_step: [neuron_id1, neuron_id2,...]}
        for j in range(n_input_features):
            spike_time = raw_data[i, j, 0]
            neuron_id = int(raw_data[i, j, 1])
            # Ensure neuron_id is within valid range [0, n_input_features-1]
            if 0 <= neuron_id < n_input_features:
                if spike_time > 0: # Ignore potential zero padding or invalid times
                    time_step = math.floor(spike_time / DT)
                    if time_step < SIM_DURATION_STEPS:
                        if time_step not in sample_spikes:
                            sample_spikes[time_step] = []
                        sample_spikes[time_step].append(neuron_id)
            # else: # Optional: Print warning for invalid neuron IDs
            #     print(f"Warning: Sample {i}, Feature {j} has invalid neuron ID {neuron_id}")
        input_spike_trains[i] = sample_spikes # Store the dict for this sample

except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data_1D_manifold_2_classes' directory exists.")
    exit()
except ValueError as ve:
    print(f"Data Loading/Processing Error: {ve}")
    exit()

# === Split data ===
# Split the preprocessed spike trains and labels
x_train_spikes, x_test_spikes, y_train, y_test = train_test_split(input_spike_trains, labels, test_size=0.2, random_state=SEED)
n_samples_train = len(x_train_spikes)
n_samples_test = len(x_test_spikes)

# === Generate Random Input Weight Matrix ===
print(f"Generating random input weight matrix W_in ({n_input_features} x {N_NODES})...")
# Simple dense random matrix for input connections
W_in = np.random.randn(n_input_features, N_NODES) * 0.1 # Scale down initial input weights

# === LIF Neuron Parameters ===
tau_mem = 20.0    # Membrane time constant (ms)
tau_syn = 5.0     # Synaptic current time constant (ms)
v_thresh = -50.0  # Spike threshold (mV)
v_reset = -65.0   # Reset potential (mV)
v_rest = -65.0    # Resting potential (mV)
cm = 1.0          # Membrane capacitance (arbitrary units)
i_offset = 0.0    # Constant background current (nA)
t_ref = 5.0       # Refractory period (ms)
input_weight_scale = 25.0 # Scaling factor for input spike effect

# === SNN Simulation Function ===
def simulate_lif_reservoir(list_of_sample_spike_dicts, w_rec, w_in, n_neurons, sim_duration_steps):
    n_samples = len(list_of_sample_spike_dicts)
    n_inputs = w_in.shape[0]

    # Initialize states for all samples
    v_mem = np.full((n_samples, n_neurons), v_rest)
    i_syn_rec = np.zeros((n_samples, n_neurons))
    i_syn_ext = np.zeros((n_samples, n_neurons))
    ref_count = np.zeros((n_samples, n_neurons))
    spikes_out_count = np.zeros((n_samples, n_neurons)) # Store total spike counts

    print(f"Simulating SNN for {n_samples} samples over {sim_duration_steps} timesteps...")
    for t in tqdm(range(sim_duration_steps)):
        # Decay currents and voltage (vectorized for all samples)
        # Order: Decay, Update Vm based on *previous* currents, Check Spikes, Calculate new currents, Reset Vm/Refractory
        i_syn_rec *= np.exp(-DT / tau_syn)
        i_syn_ext *= np.exp(-DT / tau_syn)
        
        # Update membrane potential based on decayed currents from previous step
        v_mem += (-(v_mem - v_rest) / tau_mem + (i_syn_rec + i_syn_ext + i_offset) / cm) * DT

        # Process refractory period (vectorized)
        ref_mask = ref_count > 0
        v_mem[ref_mask] = v_reset # Keep voltage clamped during refractory
        ref_count[ref_mask] -= DT
        ref_count[ref_count < 0] = 0 # Ensure non-negative
        active_mask = ~ref_mask # Neurons not in refractory period

        # Check for reservoir spikes based on updated potential for active neurons
        spiked_mask = (v_mem >= v_thresh) & active_mask
        spikes_out_count[spiked_mask] += 1

        # --- Calculate and add new currents for *next* step based on spikes --- 
        
        # Reset external current contribution for this step
        i_syn_ext_update = np.zeros((n_samples, n_neurons))
        # Reset recurrent current contribution for this step
        i_syn_rec_update = np.zeros((n_samples, n_neurons))

        # Process external inputs for this time step t
        for sample_idx in range(n_samples):
            spikes_at_t = list_of_sample_spike_dicts[sample_idx].get(t, [])
            if spikes_at_t:
                # Sum contributions from all input neurons spiking at time t
                # W_in shape: (n_inputs, n_neurons)
                # Ensure spikes_at_t are valid indices for W_in
                valid_input_spikes = [idx for idx in spikes_at_t if 0 <= idx < n_inputs]
                if valid_input_spikes:
                     input_current = w_in[valid_input_spikes, :].sum(axis=0) * input_weight_scale
                     i_syn_ext_update[sample_idx] += input_current
        
            # Calculate recurrent current based on reservoir spikes *from this step*
            spiked_indices = np.where(spiked_mask[sample_idx])[0]
            if len(spiked_indices) > 0:
                # Sum contributions from all presynaptic reservoir neurons that spiked
                # W_rec shape: (n_neurons, n_neurons)
                recurrent_current = w_rec[:, spiked_indices].sum(axis=1)
                i_syn_rec_update[sample_idx] += recurrent_current

        # Add the newly calculated currents (will be used in the *next* iteration's decay/update)
        i_syn_ext += i_syn_ext_update
        i_syn_rec += i_syn_rec_update

        # Reset voltage and set refractory period for neurons that spiked
        v_mem[spiked_mask] = v_reset
        ref_count[spiked_mask] = t_ref

    return spikes_out_count # Return total spike counts

# === Run Simulation ===
rs_train = simulate_lif_reservoir(x_train_spikes, W_rec, W_in, N_NODES, SIM_DURATION_STEPS)
rs_test = simulate_lif_reservoir(x_test_spikes, W_rec, W_in, N_NODES, SIM_DURATION_STEPS)

# === Train Classifier ===
clf = RidgeClassifier(alpha=1.0)
print("Training classifier (Random SNN Reservoir)...")
# Use spike counts as features
clf.fit(rs_train, y_train)
y_pred = clf.predict(rs_test)

# === Evaluation ===
print("\n===== Evaluation Summary (Random SNN Reservoir) =====")
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
output_dir = "figs_random_connected_snn_10cls"
os.makedirs(output_dir, exist_ok=True)
# np.save(os.path.join(output_dir, "rs_train_snn.npy"), rs_train) # Optional: Save if needed
# np.save(os.path.join(output_dir, "rs_test_snn.npy"), rs_test) # Optional: Save if needed
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(output_dir, "predictions_snn.csv"), index=False)

# Save classification report to CSV
report_df = pd.DataFrame(class_report).transpose()
report_csv_path = os.path.join(output_dir, "classification_report_snn.csv")
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

print(f"Output files will be saved to directory: {output_dir}")

# Plot Confusion Matrix
unique_labels_sorted = sorted(np.unique(np.concatenate((y_train, y_test))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels_sorted,
            yticklabels=unique_labels_sorted)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Random SNN, 10 Classes)')
cm_save_path = os.path.join(output_dir, "confusion_matrix_snn.png")
plt.savefig(cm_save_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_save_path}")

# === PCA visualization of reservoir states (spike counts) ===
print("Generating PCA visualization of reservoir states (Random SNN)...")
pca = PCA(n_components=2)
# Check if rs_train/rs_test have non-zero variance
if np.any(np.std(rs_train, axis=0) > 1e-6):
    rs_train_2d = pca.fit_transform(rs_train)
    if np.any(np.std(rs_test, axis=0) > 1e-6): # Check test set variance too
        rs_test_2d = pca.transform(rs_test)
    else:
        print("Warning: Test set reservoir states have zero variance. PCA transform might fail or be meaningless.")
        # Handle case with zero variance in test set (e.g., set test points to origin or skip plotting them)
        rs_test_2d = np.zeros((rs_test.shape[0], 2))

    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance (Random SNN): PC1 = {explained_var[0]:.2f}, PC2 = {explained_var[1]:.2f}, Total = {sum(explained_var[:2]):.2f}")

    plt.figure(figsize=(12, 10))
    unique_labels_train = np.unique(y_train)
    unique_labels_test = np.unique(y_test)

    # Plot training data
    for label in unique_labels_train:
        indices = np.where(y_train == label)[0]
        plt.scatter(rs_train_2d[indices, 0], rs_train_2d[indices, 1],
                   label=f'Train Class {label}', alpha=0.7, s=50)

    # Plot testing data
    for label in unique_labels_test:
        indices = np.where(y_test == label)[0]
        if indices.size > 0: # Check if there are test points for this label
             plt.scatter(rs_test_2d[indices, 0], rs_test_2d[indices, 1],
                   label=f'Test Class {label}', alpha=0.3, s=25, marker='x')

    plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
    plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
    plt.title('PCA Projection of Reservoir States (Spike Counts - Random SNN, 10 Classes)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    pca_save_path = os.path.join(output_dir, "reservoir_states_pca_snn.png")
    plt.savefig(pca_save_path)
    plt.close()
    print(f"PCA projection saved to {pca_save_path}")
else:
    print("PCA visualization skipped: Training reservoir states (spike counts) have zero variance.")

print("Script finished.")
