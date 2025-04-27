import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conn2res.connectivity import Conn # Import Conn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA # 添加 PCA 导入
# from sklearn.preprocessing import StandardScaler # Not used in this evaluation loop
import itertools
import time
from functools import reduce # Needed for combining multiple node sets

# === Setup ===
SEED = 42
np.random.seed(SEED)
REGIONS_TO_TEST = ['DA', 'VIS']  # 只测试DA+VIS组合
TARGET_SPECTRAL_RADIUS = 0.95
INPUT_SCALING = 1.0
LEAKING_RATE = 0.3
TEST_SIZE = 0.2

# === Load Connectome ===
# Set environment variable if needed, assuming 'human' directory is relative
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

# === Load data ===
# Assuming data is in the same directory or accessible path
try:
    start_data_load = time.time()
    # Load data once
    data_full = np.load("data_1D_manifold_2_classes_more_samples/spike_data.npy")[:, :, 0]
    labels_full = np.load("data_1D_manifold_2_classes_more_samples/spike_labels.npy")
    print(f"Data loaded ({data_full.shape[0]} samples) in {time.time() - start_data_load:.2f}s.")
except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data_1D_manifold_2_classes_more_samples' directory exists and contains the .npy files.")
    exit()

# === Evaluation Function ===
def evaluate_regions(target_regions, conn_full, data_full, labels_full, seed):
    """Evaluates ESN performance for a given set of target regions (1 to MAX_COMBINATION_SIZE)."""
    eval_start_time = time.time()
    region_name = "+".join(sorted(target_regions)) # Sort for consistent naming
    print(f"--- Evaluating Region(s): {region_name} ({len(target_regions)} regions) ---")

    try:
        # --- Get Nodes ---
        # Get nodes for each region and combine them
        node_sets = [conn_full.get_nodes(r) for r in target_regions]
        # Use reduce with np.union1d to combine all node sets
        region_nodes = reduce(np.union1d, node_sets).astype(int)

        if len(region_nodes) == 0:
            print(f"Error: No nodes found for region combination '{region_name}'. Skipping.")
            return region_name, None, None, 0.0, None, None, None, None, None, None, None, None, None # Return None for metrics

        N_RESERVOIR = len(region_nodes)
        if N_RESERVOIR <= 1: # Need at least 2 nodes for a meaningful reservoir
             print(f"Warning: Only {N_RESERVOIR} node(s) found for region combination '{region_name}'. Skipping.")
             return region_name, None, N_RESERVOIR, 0.0, None, None, None, None, None, None, None, None, None # Return None for metrics

        print(f"Using {N_RESERVOIR} nodes for {region_name}.")

        # --- ESN Parameters & Matrices ---
        n_inputs = data_full.shape[1] # Get input dimension from full data
        np.random.seed(seed) # Reset seed for consistent W_in generation per evaluation
        W_in = (np.random.rand(N_RESERVOIR, n_inputs) * 2 - 1) * INPUT_SCALING
        W = conn_full.w[np.ix_(region_nodes, region_nodes)].copy()

        # Scale spectral radius
        try:
            eigenvalues = np.linalg.eigvals(W)
            current_spectral_radius = np.max(np.abs(eigenvalues))
            if current_spectral_radius > 1e-9:
                W = W * (TARGET_SPECTRAL_RADIUS / current_spectral_radius)
            else:
                print(f"Warning: Spectral radius near zero for {region_name}. Using unscaled W.")
        except np.linalg.LinAlgError:
             print(f"Warning: Eigenvalue computation failed for {region_name}. Using unscaled W.")


        # --- Split data ---
        x_train, x_test, y_train, y_test = train_test_split(
            data_full, labels_full, test_size=TEST_SIZE, random_state=seed
        )
        n_train_samples = x_train.shape[0]
        n_test_samples = x_test.shape[0]

        # --- Simulate ESN ---
        # Define run_esn locally to capture W, W_in etc.
        def run_esn(input_data, n_samples, W_sim, Win_sim, N_res_sim, leak_rate_sim):
            states = np.zeros((n_samples, N_res_sim))
            x = np.zeros(N_res_sim)
            for t in range(n_samples):
                u = input_data[t, :]
                x_pre_activation = Win_sim @ u + W_sim @ x
                x = (1 - leak_rate_sim) * x + leak_rate_sim * np.tanh(x_pre_activation)
                states[t, :] = x
            return states

        rs_train = run_esn(x_train, n_train_samples, W, W_in, N_RESERVOIR, LEAKING_RATE)
        rs_test = run_esn(x_test, n_test_samples, W, W_in, N_RESERVOIR, LEAKING_RATE)

        # --- Train & Evaluate Classifier ---
        clf = RidgeClassifier(alpha=1.0, random_state=seed)
        clf.fit(rs_train, y_train)
        y_pred = clf.predict(rs_test)
        
        # 计算各种性能指标
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 生成分类报告
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Accuracy for {region_name}: {accuracy:.4f}")
        print(f"Macro F1 for {region_name}: {macro_f1:.4f}")
        print(f"Micro F1 for {region_name}: {micro_f1:.4f}")
        print(f"Weighted F1 for {region_name}: {weighted_f1:.4f}")

        eval_duration = time.time() - eval_start_time
        print(f"--- Finished {region_name} in {eval_duration:.2f}s ---")
        # 返回 rs_train, rs_test, y_train, y_test 用于 PCA 可视化
        return region_name, accuracy, N_RESERVOIR, eval_duration, macro_f1, micro_f1, weighted_f1, cm, class_report, rs_train, rs_test, y_train, y_test

    except ValueError as e:
        print(f"Error processing region combination '{region_name}': {e}. Skipping.")
        return region_name, None, None, time.time() - eval_start_time, None, None, None, None, None, None, None, None, None # Return None for metrics
    except Exception as e_gen:
         print(f"Unexpected error processing region combination '{region_name}': {e_gen}. Skipping.")
         return region_name, None, None, time.time() - eval_start_time, None, None, None, None, None, None, None, None, None # Return None for metrics

# === Main Evaluation ===
results = {}
total_eval_time = 0

print(f"===== Evaluating DA+VIS Region Combination =====")
# 测试DA+VIS组合
region_combo = ['DA', 'VIS']
# 接收 evaluate_regions 返回的额外数据用于 PCA
name, acc, n_nodes, duration, macro_f1, micro_f1, weighted_f1, confusion_mat, class_report, rs_train_pca, rs_test_pca, y_train_pca, y_test_pca = evaluate_regions(
    region_combo, conn_full, data_full, labels_full, SEED)

if acc is not None:
    # 将 PCA 相关数据暂存，避免存入 results 字典
    results[name] = (acc, n_nodes, macro_f1, micro_f1, weighted_f1, confusion_mat, class_report)
total_eval_time += duration

# === Report Results ===
print("===== Evaluation Summary =====")
if not results:
    print("No results were obtained.")
    exit()

# 输出详细结果 (因为只有一个结果，直接处理)
if name in results:
    acc, n_nodes, macro_f1, micro_f1, weighted_f1, cm, report = results[name]
    print(f"Region Configuration: {name}")
    print(f"Reservoir Size: {n_nodes}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    # 使用 pandas DataFrame 输出分类报告，格式更整齐
    report_df_display = pd.DataFrame(report).transpose()
    print(report_df_display.to_string(float_format="{:.4f}".format))
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(labels_full)), 
                yticklabels=sorted(set(labels_full)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {name}')
    cm_filename = f'confusion_matrix_{name.replace("+", "_")}.png'
    plt.savefig(cm_filename)
    print(f"\nConfusion matrix saved as '{cm_filename}'")
    plt.close() # 关闭图形，避免后续 PCA 图重叠
    
    # 保存详细的报告到CSV文件
    report_df = pd.DataFrame(report).transpose()
    report_filename = f'classification_report_{name.replace("+", "_")}.csv'
    report_df.to_csv(report_filename)
    print(f"Classification report saved as '{report_filename}'")

    # === PCA visualization ===
    if rs_train_pca is not None and rs_test_pca is not None:
        print("\nGenerating PCA visualization of reservoir states...")
        pca = PCA(n_components=2)
        try:
            rs_train_2d = pca.fit_transform(rs_train_pca)
            rs_test_2d = pca.transform(rs_test_pca)

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
                plt.scatter(rs_test_2d[indices, 0], rs_test_2d[indices, 1],
                           label=f'Test Class {label}', alpha=0.3, s=25, marker='x')

            plt.xlabel(f'PC1 ({explained_var[0]:.2f})')
            plt.ylabel(f'PC2 ({explained_var[1]:.2f})')
            plt.title(f'PCA Projection of Reservoir States ({name})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pca_filename = f'pca_projection_{name.replace("+", "_")}.png'
            plt.savefig(pca_filename)
            print(f"PCA projection saved as '{pca_filename}'")
            plt.close()
        except Exception as pca_e:
            print(f"Error during PCA visualization: {pca_e}")
    else:
        print("\nSkipping PCA visualization because reservoir states are not available.")


print("---------------------------------")
print(f"Total evaluation time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")
print("Script finished.")
