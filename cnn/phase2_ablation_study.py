import os
import sys
import json
import math
import numpy as np
import pandas as pd

# Set local KERAS_HOME within workspace to avoid sandbox permission errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.environ.setdefault("KERAS_HOME", os.path.join(REPO_DIR, ".keras"))

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Permute
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from numpy.lib.stride_tricks import as_strided

# -------------------------------------------------------------
# Configuration (Relative Paths)
# -------------------------------------------------------------
DATASET_DIR = os.path.join(REPO_DIR, "dataset", "phase2")
CNN_DIR = SCRIPT_DIR
DT = 60
VECTOR_LENGTH = 32 * 32
NUM_CLASSES = 4
CLASSES = np.array(['benign', 'sysrv', 'xmrig', 'mirai'])

# Load exact 299 syscalls master list from notebook
nb_path = os.path.join(CNN_DIR, "phase2_60sec_CNN.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)
local_vars = {}
exec("".join(nb["cells"][2]["source"]), {}, local_vars)
syscalls = local_vars["syscalls"]

# Global Master Label Encoder
global_label_encoder = LabelEncoder()
global_label_encoder.fit(syscalls)
TOTAL_SYSCALLS = len(global_label_encoder.classes_)
print(f"Global Master Syscall Table initialized with {TOTAL_SYSCALLS} syscalls.")

# -------------------------------------------------------------
# Data Loader (Step 1: Clean Global Encoding, 1-based, 0=PAD)
# -------------------------------------------------------------
def load_all_data():
    """
    Loads Phase 2 60sec dataset:
    - nrows=1024 fast loading (~5-6s)
    - Global Master Tokenizer with 1-based indexing (0 is reserved for PAD)
    - Normalized by / 300.0
    """
    classes = [f"0/{DT}sec_0", f"1/{DT}sec_1", f"2/{DT}sec_2", f"3/{DT}sec_3"]
    file_paths = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.csv'):
                file_paths.append((os.path.join(class_dir, file_name), class_idx))

    x_list, y_list = [], []
    for fp, c_idx in file_paths:
        try:
            data = pd.read_csv(fp, nrows=1024, encoding='utf-8')
        except UnicodeDecodeError:
            data = pd.read_csv(fp, nrows=1024, encoding='ISO-8859-1')
        
        vector = np.zeros(VECTOR_LENGTH, dtype=np.float32)
        encoded = global_label_encoder.transform(data['SYSTEM_CALL']) + 1
        n = min(len(encoded), VECTOR_LENGTH)
        vector[:n] = encoded[:n] / 300.0
            
        x_list.append(vector)
        y_list.append(c_idx)

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

# -------------------------------------------------------------
# CNN Model Builder
# -------------------------------------------------------------
def build_1d_cnn():
    inp = Input(shape=(VECTOR_LENGTH, 1))
    x = Conv1D(16, kernel_size=3, strides=10, padding='valid', activation='relu')(inp)
    x = Conv1D(32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = Conv1D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = Permute((2, 1))(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
    out = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(1e-4))(x)
    model = Model(inp, out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

# -------------------------------------------------------------
# Quantization Helpers
# -------------------------------------------------------------
def to_q15(x):
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int64)

def to_q15_bias(x):
    return np.clip(np.round(x * 32768.0), -2**31, 2**31 - 1).astype(np.int64)

def quantize_weights(model):
    conv_layers = [l for l in model.layers if 'conv1d' in l.name]
    dense_layers = [l for l in model.layers if l.name.startswith('dense')]
    c1_w_f, c1_b_f = conv_layers[0].get_weights()
    c2_w_f, c2_b_f = conv_layers[1].get_weights()
    c3_w_f, c3_b_f = conv_layers[2].get_weights()
    d_w_f, d_b_f = dense_layers[0].get_weights()
    fc_w_f, fc_b_f = dense_layers[1].get_weights()
    
    c1_w = np.transpose(c1_w_f[:, 0, :], (1, 0))
    c2_w = np.transpose(c2_w_f, (2, 1, 0))
    c3_w = np.transpose(c3_w_f, (2, 1, 0))
    d_w = d_w_f.T
    fc_w = fc_w_f.T
    
    out = {}
    for name, w, b in [('conv1', c1_w, c1_b_f), ('conv2', c2_w, c2_b_f), ('conv3', c3_w, c3_b_f),
                        ('dense', d_w, d_b_f), ('fc2', fc_w, fc_b_f)]:
        out[f'{name}_w'] = to_q15(w)
        out[f'{name}_b'] = to_q15_bias(b)
    return out

def windows_1d(a, out_len, k, stride, axis):
    a = np.ascontiguousarray(a)
    shape = list(a.shape); shape[axis] = out_len; shape = shape + [k]
    strides = list(a.strides); ts = strides[axis]
    strides[axis] = ts * stride; strides = strides + [ts]
    return as_strided(a, shape=shape, strides=strides)

# eBPF Round-to-Nearest bit-shift: (accum + (1 << (s - 1))) >> s
def r_shift(val, s):
    if s <= 0:
        return val << (-s)
    return (val + (1 << (s - 1))) >> s

# eBPF Floor bit-shift: val >> s
def f_shift(val, s):
    if s <= 0:
        return val << (-s)
    return val >> s

# Universal eBPF Pure Integer Inference Engine
def ebpf_q15_forward(qw, X, use_round=True, dense_shift=18, s1=10, s2=1, s3=1, seq_len=1024):
    """
    Universal Pure Integer Inference Engine:
    - use_round=True: Round-to-Nearest shift (Step 2)
    - dense_shift=18: Saturation Prevention dynamic shift (Step 3: >> 18, bias >> 3, fc >> 12)
    - dense_shift=15: Standard author Q15 shift (No Saturation Prevention: >> 15, fc >> 15)
    """
    c1w, c1b = qw['conv1_w'], qw['conv1_b']
    c2w, c2b = qw['conv2_w'], qw['conv2_b']
    c3w, c3b = qw['conv3_w'], qw['conv3_b']
    dw, db = qw['dense_w'], qw['dense_b']
    fcw, fcb = qw['fc2_w'], qw['fc2_b']
    
    shift_op = r_shift if use_round else f_shift
    
    # Dynamic scale compensation parameters
    delta_shift = dense_shift - 15
    fc_shift = 15 - delta_shift
    
    F1, K1 = c1w.shape; F2, _, K2 = c2w.shape; F3, _, K3 = c3w.shape
    c1out = (seq_len - K1) // s1 + 1
    c2out = (c1out - K2) // s2 + 1
    c3out = (c2out - K3) // s3 + 1
    
    # 1. Quantize input vector to Q15 integers
    Xq = np.clip(np.round(X * 32768.0), -32768, 32767).astype(np.int64)
    
    # Conv1
    win1 = windows_1d(Xq, c1out, K1, s1, axis=1)
    a1 = np.maximum(0, shift_op(np.einsum('ntk,fk->nft', win1, c1w), 15) + c1b[None, :, None])
    a1c = np.clip(a1, -32768, 32767)
    
    # Conv2
    win2 = windows_1d(a1c, c2out, K2, s2, axis=2)
    a2 = np.maximum(0, shift_op(np.einsum('nctk,fck->nft', win2, c2w), 15) + c2b[None, :, None])
    a2c = np.clip(a2, -32768, 32767)
    
    # Conv3
    win3 = windows_1d(a2c, c3out, K3, s3, axis=2)
    a3 = np.maximum(0, shift_op(np.einsum('nctk,fck->nft', win3, c3w), 15) + c3b[None, :, None])
    a3c = np.clip(a3, -32768, 32767)
    
    # Flatten & Dense1
    N = X.shape[0]
    flat = a3c.reshape(N, F3 * c3out)
    bias_d = (db >> delta_shift) if delta_shift > 0 else db
    hidden = np.clip(np.maximum(0, shift_op(flat @ dw.T, dense_shift) + bias_d[None, :]), -32768, 32767)
    
    # Dense2 (Softmax Logits)
    logits = shift_op(hidden @ fcw.T, fc_shift) + fcb[None, :]
    return np.argmax(logits, axis=1)


# =============================================================
# MAIN EXECUTION
# =============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("STEP-BY-STEP ABLATION STUDY: Phase 2 60sec")
    print("Target: Solve Quantization Accuracy Collapse for eBPF Deployment")
    print("=" * 80)
    
    results = {}
    
    # Historical Baselines on Corrupt Data (for context)
    results['[Baseline Ref] Corrupt Data (Float32)'] = (0.8592, None, None)
    results['[Baseline Ref] Corrupt Data (Orig Q15 Collapse)'] = (0.5102, None, None)
    
    # 1. Load Clean Dataset (Step 1)
    print("\n>>> Loading Clean Dataset (Global Master Tokenizer, 1-based, 0=PAD)...")
    X, y = load_all_data()
    print(f"Loaded {len(X)} samples successfully.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_test_3d = X_test.reshape(-1, VECTOR_LENGTH, 1)
    
    # 2. Load or Train High-Accuracy Model
    model_path = os.path.join(CNN_DIR, "phase2_60sec.h5")
    if os.path.exists(model_path):
        print(f"\n>>> Loading Model from: {model_path}")
        model = load_model(model_path)
    else:
        print(f"\n>>> Model not found at {model_path}. Training new model...")
        classes_u = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes_u, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes_u, weights)}
        
        X_train_3d = X_train.reshape(-1, VECTOR_LENGTH, 1)
        X_val_3d = X_val.reshape(-1, VECTOR_LENGTH, 1)
        model = build_1d_cnn()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001)
        checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        model.fit(X_train_3d, y_train, epochs=80, batch_size=32, validation_data=(X_val_3d, y_val),
                  class_weight=class_weight, callbacks=[reduce_lr, checkpoint], verbose=1)
        model = load_model(model_path)

    qw = quantize_weights(model)
    
    # ---------------------------------------------------------
    # ABLATION EXPERIMENTS ON CLEAN RETRAINED MODEL
    # ---------------------------------------------------------
    
    # [1] Float32 Accuracy
    print("\n--- 1. Clean Retrained Model (Float32) ---")
    y_pred_f32 = np.argmax(model.predict(X_test_3d, verbose=0), axis=1)
    acc_f32 = accuracy_score(y_test, y_pred_f32)
    results['1. Step 1 (Retrained Float32)'] = (acc_f32, y_pred_f32, y_test)
    print(f"Float32 Accuracy: {acc_f32*100:.2f}%")
    
    # [2] Step 1 Only: Clean Data + Original Q15 (Floor Shift >> 15, Saturation Present)
    print("\n--- 2. Step 1 Only: Clean Data + Original Q15 Engine (Floor Shift >> 15) ---")
    y_pred_s1_orig = ebpf_q15_forward(qw, X_test, use_round=False, dense_shift=15)
    acc_s1_orig = accuracy_score(y_test, y_pred_s1_orig)
    results['2. Step 1 Only (Orig Q15 >>15, Floor)'] = (acc_s1_orig, y_pred_s1_orig, y_test)
    print(f"Step 1 Only Accuracy: {acc_s1_orig*100:.2f}%")
    
    # [3] Step 1 + Step 2: Clean Data + Round-to-Nearest Shift (dense_shift=15)
    print("\n--- 3. Step 1 + Step 2: Round-to-Nearest Bit Shift (dense_shift=15) ---")
    y_pred_s1_s2 = ebpf_q15_forward(qw, X_test, use_round=True, dense_shift=15)
    acc_s1_s2 = accuracy_score(y_test, y_pred_s1_s2)
    results['3. Step 1+2 (Rounding Shift >>15)'] = (acc_s1_s2, y_pred_s1_s2, y_test)
    print(f"Step 1+2 Accuracy: {acc_s1_s2*100:.2f}%")
    
    # [4] Step 1 + Step 3: Clean Data + Dynamic Shift >> 18 (Floor Shift)
    print("\n--- 4. Step 1 + Step 3: Dynamic Shift >> 18 (Floor Shift, Saturation Solved) ---")
    y_pred_s1_s3 = ebpf_q15_forward(qw, X_test, use_round=False, dense_shift=18)
    acc_s1_s3 = accuracy_score(y_test, y_pred_s1_s3)
    results['4. Step 1+3 (Dynamic Shift >>18, Floor)'] = (acc_s1_s3, y_pred_s1_s3, y_test)
    print(f"Step 1+3 Accuracy: {acc_s1_s3*100:.2f}%")
    
    # [5] COMBINED (All Steps): eBPF Integer Candidate (Round + Dynamic Shift >> 18)
    print("\n--- 5. COMBINED: All Steps (eBPF Pure Integer Candidate: Round + Shift 18) ---")
    y_pred_comb = ebpf_q15_forward(qw, X_test, use_round=True, dense_shift=18)
    acc_comb = accuracy_score(y_test, y_pred_comb)
    results['5. COMBINED (All Steps: Final eBPF Engine)'] = (acc_comb, y_pred_comb, y_test)
    print(f"COMBINED Accuracy: {acc_comb*100:.2f}%")
    
    # ---------------------------------------------------------
    # Summary Table & Report Generation
    # ---------------------------------------------------------
    print("\n" + "=" * 95)
    print("FINAL ABLATION STUDY COMPARISON TABLE (Phase 2 60sec)")
    print("=" * 95)
    
    summary_rows = []
    for exp_name, (acc, y_pred, y_true) in results.items():
        if y_pred is not None and y_true is not None:
            rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
            row = {
                'Experiment': exp_name,
                'Accuracy': f"{acc*100:.2f}%",
                'Benign Recall': f"{rep['benign']['recall']*100:.1f}%",
                'Sysrv Recall': f"{rep['sysrv']['recall']*100:.1f}%",
                'XMRig Recall': f"{rep['xmrig']['recall']*100:.1f}%",
                'Mirai Recall': f"{rep['mirai']['recall']*100:.1f}%",
                'Macro F1': f"{rep['macro avg']['f1-score']*100:.2f}%"
            }
        else:
            # Baseline reference rows
            row = {
                'Experiment': exp_name,
                'Accuracy': f"{acc*100:.2f}%",
                'Benign Recall': "6.1%" if "Collapse" in exp_name else "93.4%",
                'Sysrv Recall': "80.0%" if "Collapse" in exp_name else "52.7%",
                'XMRig Recall': "95.9%" if "Collapse" in exp_name else "95.3%",
                'Mirai Recall': "89.7%" if "Collapse" in exp_name else "91.0%",
                'Macro F1': "56.86%" if "Collapse" in exp_name else "82.45%"
            }
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    
    print("\n--- Confusion Matrix: COMBINED Final eBPF Engine ---")
    cm_comb = confusion_matrix(y_test, y_pred_comb)
    print(pd.DataFrame(cm_comb, index=CLASSES, columns=CLASSES))
    
    # Save results to CSV in cnn/
    out_csv = os.path.join(CNN_DIR, "phase2_60sec_ablation_results.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved complete ablation study results to: {out_csv}")
