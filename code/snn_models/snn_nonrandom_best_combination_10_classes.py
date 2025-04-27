import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conn2res.connectivity import Conn # Import Conn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix 
import seaborn as sns 
from sklearn.decomposition import PCA 
import itertools
import time
from functools import reduce # Needed for combining multiple node sets
from tqdm import tqdm # For progress bar
import math

# === Setup ===
SEED = 42
np.random.seed(SEED)
REGION_TO_TEST = ['DMN'] # <<<--- Test only DMN
# MAX_COMBINATION_SIZE = 3 # Removed
TARGET_SPECTRAL_RADIUS = 0.95 # Target for scaling W_rec
DT = 1.0 # Simulation time step (ms)
OUTPUT_DIR = "figs_snn_nonrandom_DMN_10cls" # <<<--- Updated output directory for 10 classes
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# === Load Connectome ===
os.environ["CONN2RES_DATA"] = os.path.abspath("human")
CONNECTOME_LOADED = False
try:
    start_conn_load = time.time()
    conn_full = Conn(subj_id=0) # Load the full connectome
    print(f"Full connectome loaded with {conn_full.w.shape[0]} nodes in {time.time() - start_conn_load:.2f}s.")
    CONNECTOME_LOADED = True
except Exception as e:
    print(f"Error loading connectome: {e}")
    print("Cannot proceed without connectome.")
    exit() # Exit if connectome cannot be loaded

# === Load and Preprocess Spike Data ===
print("Loading and preprocessing spike event data...")
try:
    raw_data = np.load("data_2D_manifold_10_classes/1000 samples/spike_data.npy")
    labels_full = np.load("data_2D_manifold_10_classes/1000 samples/spike_labels.npy")

    if raw_data.ndim != 3 or raw_data.shape[2] != 2:
        raise ValueError(f"Loaded data has unexpected shape: {raw_data.shape}. Expected (samples, features, 2).")

    n_samples, n_input_features, _ = raw_data.shape
    print(f"Loaded raw data shape: {raw_data.shape} (samples, features, [time, id])")
    print(f"Number of input features: {n_input_features}")
    print(f"Loaded labels shape: {labels_full.shape}")

    max_spike_time = np.nanmax(raw_data[:, :, 0])
    SIM_DURATION_STEPS = math.ceil(max_spike_time / DT) + 10
    print(f"Max spike time found: {max_spike_time:.2f} ms. Simulation duration: {SIM_DURATION_STEPS} steps ({SIM_DURATION_STEPS * DT} ms)")

    print("Preprocessing spike data...")
    input_spike_trains = [{} for _ in range(n_samples)]
    for i in tqdm(range(n_samples)):
        sample_spikes = {}
        for j in range(raw_data.shape[1]):
            spike_time = raw_data[i, j, 0]
            neuron_id = int(raw_data[i, j, 1])
            if not np.isnan(spike_time) and spike_time > 0 and 0 <= neuron_id < n_input_features:
                time_step = math.floor(spike_time / DT)
                if time_step < SIM_DURATION_STEPS:
                    if time_step not in sample_spikes:
                        sample_spikes[time_step] = []
                    sample_spikes[time_step].append(neuron_id)
        input_spike_trains[i] = sample_spikes

except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data_2D_manifold_10_classes/1000 samples/' directory exists.")
    exit()
except ValueError as ve:
    print(f"Data Loading/Processing Error: {ve}")
    exit()

# === Split Data (Preprocessed Spike Trains) ===
x_train_spikes, x_test_spikes, y_train_split, y_test_split = train_test_split(
    input_spike_trains, labels_full, test_size=0.2, random_state=SEED
)
n_samples_train = len(x_train_spikes)
n_samples_test = len(x_test_spikes)
print(f"Data split: {n_samples_train} train samples, {n_samples_test} test samples.")

# === SNN Simulation Function ===
def simulate_lif_reservoir(list_of_sample_spike_dicts, w_rec, w_in, n_neurons, sim_duration_steps, lif_params):
    n_samples = len(list_of_sample_spike_dicts)
    n_inputs = w_in.shape[0]
    v_rest, v_reset, v_thresh, cm, tau_mem, tau_syn, t_ref, i_offset, input_scale = lif_params

    v_mem = np.full((n_samples, n_neurons), v_rest)
    i_syn_rec = np.zeros((n_samples, n_neurons))
    i_syn_ext = np.zeros((n_samples, n_neurons))
    ref_count = np.zeros((n_samples, n_neurons))
    spikes_out_count = np.zeros((n_samples, n_neurons))

    for t in range(sim_duration_steps):
        i_syn_rec *= np.exp(-DT / tau_syn)
        i_syn_ext *= np.exp(-DT / tau_syn)
        dv = (-(v_mem - v_rest) / tau_mem + (i_syn_rec + i_syn_ext + i_offset) / cm) * DT
        v_mem += dv
        ref_mask = ref_count > 0
        v_mem[ref_mask] = v_reset
        ref_count[ref_mask] -= DT
        active_mask = ~ref_mask
        spiked_mask = (v_mem >= v_thresh) & active_mask
        spikes_out_count[spiked_mask] += 1
        i_syn_ext_update = np.zeros((n_samples, n_neurons))
        i_syn_rec_update = np.zeros((n_samples, n_neurons))
        for sample_idx in range(n_samples):
            spikes_at_t = list_of_sample_spike_dicts[sample_idx].get(t, [])
            if spikes_at_t:
                valid_input_spikes = [idx for idx in spikes_at_t if 0 <= idx < n_inputs]
                if valid_input_spikes:
                     input_current = w_in[valid_input_spikes, :].sum(axis=0) * input_scale
                     i_syn_ext_update[sample_idx] += input_current
            spiked_indices = np.where(spiked_mask[sample_idx])[0]
            if len(spiked_indices) > 0:
                recurrent_current = w_rec[:, spiked_indices].sum(axis=1)
                i_syn_rec_update[sample_idx] += recurrent_current
        i_syn_ext += i_syn_ext_update
        i_syn_rec += i_syn_rec_update
        v_mem[spiked_mask] = v_reset
        ref_count[spiked_mask] = t_ref
    return spikes_out_count

# === Evaluation Function (Returns Full Metrics & States) ===
def evaluate_regions_snn(target_regions, conn_full, x_train_spikes, x_test_spikes, y_train, y_test, seed, lif_params):
    eval_start_time = time.time()
    region_name = "+".join(sorted(target_regions))
    print(f"--- Evaluating SNN for Region(s): {region_name} ({len(target_regions)} regions) ---")

    # Initialize return values for failure cases
    default_return = region_name, None, None, 0.0, None, None, None, None, None, None, None, None, None

    try:
        node_sets = [conn_full.get_nodes(r) for r in target_regions]
        region_nodes = reduce(np.union1d, node_sets).astype(int)

        if len(region_nodes) == 0:
            print(f"Error: No nodes found for '{region_name}'. Skipping.")
            return default_return

        N_RESERVOIR = len(region_nodes)
        if N_RESERVOIR <= 1:
             print(f"Warning: Only {N_RESERVOIR} node(s) found for '{region_name}'. Skipping.")
             # Still return N_RESERVOIR in this case, but metrics will be None
             return region_name, None, N_RESERVOIR, 0.0, None, None, None, None, None, None, None, None, None

        print(f"Using {N_RESERVOIR} nodes for {region_name}.")

        n_inputs = n_input_features
        np.random.seed(seed)
        W_in = np.random.randn(n_inputs, N_RESERVOIR) * 0.1
        W_rec = conn_full.w[np.ix_(region_nodes, region_nodes)].copy()

        try:
            eigenvalues = np.linalg.eigvals(W_rec)
            current_spectral_radius = np.max(np.abs(eigenvalues))
            if current_spectral_radius > 1e-9:
                W_rec = W_rec * (TARGET_SPECTRAL_RADIUS / current_spectral_radius)
            else:
                print(f"Warning: W_rec spectral radius near zero for {region_name}. Using unscaled.")
        except np.linalg.LinAlgError:
             print(f"Warning: Eigenvalue computation failed for W_rec of {region_name}. Using unscaled.")

        lif_params_tuple = lif_params
        print("Simulating SNN for training data...")
        rs_train = simulate_lif_reservoir(x_train_spikes, W_rec, W_in, N_RESERVOIR, SIM_DURATION_STEPS, lif_params_tuple)
        print("Simulating SNN for testing data...")
        rs_test = simulate_lif_reservoir(x_test_spikes, W_rec, W_in, N_RESERVOIR, SIM_DURATION_STEPS, lif_params_tuple)

        accuracy = 0.0
        macro_f1 = 0.0
        micro_f1 = 0.0
        weighted_f1 = 0.0
        cm = None
        class_report = {}

        if np.all(np.std(rs_train, axis=0) < 1e-6):
             print(f"Warning: Training states have zero variance for {region_name}. Metrics set to 0/None.")
             # Keep metrics as 0/None, but still return states for potential inspection
        else:
            clf = RidgeClassifier(alpha=1.0, random_state=seed)
            clf.fit(rs_train, y_train)
            y_pred = clf.predict(rs_test)
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
            weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"Accuracy for {region_name}: {accuracy:.4f}")
        print(f"Macro F1 for {region_name}: {macro_f1:.4f}") 
        eval_duration = time.time() - eval_start_time
        print(f"--- Finished {region_name} in {eval_duration:.2f}s ---")
        
        # Return full details including states and labels for PCA
        return region_name, accuracy, N_RESERVOIR, eval_duration, macro_f1, micro_f1, weighted_f1, cm, class_report, rs_train, rs_test, y_train, y_test

    except ValueError as e:
        print(f"Error processing '{region_name}': {e}. Skipping.")
        return default_return
    except Exception as e_gen:
         print(f"Unexpected error processing '{region_name}': {e_gen}. Skipping.")
         return default_return

# === Main Evaluation (Single Region: DMN) ===
print(f"===== Evaluating SNN for Single Region: {REGION_TO_TEST[0]} =====")
lif_params_eval = (v_rest, v_reset, v_thresh, cm, tau_mem, tau_syn, t_ref, i_offset, input_weight_scale)
total_eval_time = 0

# Evaluate only the specified region
eval_output = evaluate_regions_snn(
    REGION_TO_TEST, conn_full, 
    x_train_spikes, x_test_spikes, y_train_split, y_test_split, 
    SEED, lif_params_eval
)

# Unpack all results
name, acc, n_nodes, duration, macro_f1, micro_f1, weighted_f1, cm, report, rs_train, rs_test, y_train_pca, y_test_pca = eval_output
total_eval_time += duration

# === Report Results ===
print(f"===== SNN Evaluation Summary for {name} =====")

if acc is None:
    print("Evaluation failed or was skipped.")
    exit()

print(f"Region Configuration: {name}")
print(f"Nodes: {n_nodes}")
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# --- Save & Plot Confusion Matrix ---
if cm is not None:
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(10, 8)) # Adjusted size for 10 classes
    unique_labels_sorted = sorted(np.unique(labels_full)) # Get unique labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels_sorted, yticklabels=unique_labels_sorted)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (SNN: {name}, 10 Classes)') # Updated title
    # Updated filename
    cm_filename = os.path.join(OUTPUT_DIR, f'confusion_matrix_snn_{name.replace("+", "_")}_10cls.png')
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to {cm_filename}")
else:
    print("\nConfusion Matrix not available.")

# --- Save Classification Report ---
if report:
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df.to_string(float_format="{:.4f}".format))
    # Updated filename
    report_filename = os.path.join(OUTPUT_DIR, f'classification_report_snn_{name.replace("+", "_")}_10cls.csv')
    report_df.to_csv(report_filename)
    print(f"Classification report saved to {report_filename}")
else:
     print("\nClassification Report not available.")

# --- PCA Visualization ---
print("Generating PCA visualization...")
if rs_train is not None and rs_test is not None and y_train_pca is not None and y_test_pca is not None:
    if np.any(np.std(rs_train, axis=0) > 1e-6):
        pca = PCA(n_components=2)
        rs_train_2d = pca.fit_transform(rs_train)
        
        if np.any(np.std(rs_test, axis=0) > 1e-6):
             rs_test_2d = pca.transform(rs_test)
        else:
             print("Warning: Test set spike counts have zero variance. PCA transform might be meaningless.")
             rs_test_2d = np.zeros((rs_test.shape[0], 2)) # Placeholder

        explained_var = pca.explained_variance_ratio_
        print(f"PCA explained variance: PC1 = {explained_var[0]:.2f}, PC2 = {explained_var[1]:.2f}, Total = {sum(explained_var[:2]):.2f}")

        plt.figure(figsize=(12, 10))
        unique_labels_train = np.unique(y_train_pca)
        unique_labels_test = np.unique(y_test_pca)

        # Plot training data
        for label in unique_labels_train:
            indices = np.where(y_train_pca == label)[0]
            plt.scatter(rs_train_2d[indices, 0], rs_train_2d[indices, 1],
                       label=f'Train Class {label}', alpha=0.7, s=50)

        # Plot testing data
        for label in unique_labels_test:
            indices = np.where(y_test_pca == label)[0]
            if indices.size > 0:
                 plt.scatter(rs_test_2d[indices, 0], rs_test_2d[indices, 1],
                       label=f'Test Class {label}', alpha=0.3, s=25, marker='x')

        plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
        plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
        plt.title(f'PCA Projection of Reservoir States (Spike Counts - SNN: {name}, 10 Classes)') # Updated title
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
        # Updated filename
        pca_filename = os.path.join(OUTPUT_DIR, f'pca_projection_snn_{name.replace("+", "_")}_10cls.png')
        plt.savefig(pca_filename)
        print(f"PCA projection saved to {pca_filename}")
        plt.close()
    else:
        print("PCA visualization skipped: Training spike counts have zero variance.")
else:
    print("PCA visualization skipped: Missing necessary data (reservoir states or labels).")


print("\n---------------------------------")
print(f"Total evaluation time: {total_eval_time:.2f} seconds")
print(f"Results and plots saved in: {OUTPUT_DIR}")
print("Script finished.")
