# ============================================================
# QUANTUM HYBRID INTRUSION DETECTION SYSTEM (IDS)
# ============================================================
# Author: George
# Dataset: CIC-IDS-2017
# Technologies:
# - Classical Machine Learning
# - Quantum Computing (Qiskit)
# - Random Forest
# - Jensen-Shannon Distance
# ============================================================


# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    f1_score
)

from scipy.spatial.distance import jensenshannon

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from tqdm import tqdm


# ============================================================
# SECTION 2 — CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_QUBITS = 4

CHUNK_SIZE = 20000
ROWS_PER_FILE = 3000

DATASET_PATH = r"C:\Users\User\Downloads\archive (1)"

np.random.seed(RANDOM_SEED)


# ============================================================
# SECTION 3 — LOAD CSV FILES
# ============================================================

print("\n[STEP 1] Loading dataset...")

csv_files = list(Path(DATASET_PATH).glob("*.csv"))

dfs = []

for f in csv_files:

    fname = Path(f).name

    try:
        chunks_kept = []
        rows_so_far = 0

        reader = pd.read_csv(
            f,
            chunksize=CHUNK_SIZE,
            low_memory=True,
            engine="c",
            on_bad_lines="skip",
            encoding="utf-8",
            encoding_errors="ignore",
        )

        for chunk in reader:

            chunk.columns = chunk.columns.str.strip()

            # label column
            if "Label" in chunk.columns:
                label_col = "Label"
            else:
                label_col = chunk.columns[-1]

            labels_chunk = chunk[label_col].copy()

            # convert safely
            chunk = chunk.apply(pd.to_numeric, errors="coerce")

            # restore labels
            chunk[label_col] = labels_chunk

            # features only
            features = chunk.drop(
                columns=[label_col],
                errors="ignore"
            )

            # remove mostly empty columns
            features.dropna(
                axis=1,
                thresh=int(0.7 * len(features)),
                inplace=True
            )

            # fill NaNs
            features.fillna(0, inplace=True)

            # reduce memory
            for col in features.columns:
                features[col] = pd.to_numeric(
                    features[col],
                    downcast="float"
                )

            # rebuild chunk
            chunk_clean = features.copy()
            chunk_clean[label_col] = labels_chunk

            # sampling
            remaining = ROWS_PER_FILE - rows_so_far

            if remaining <= 0:
                break

            if len(chunk_clean) > remaining:
                chunk_clean = chunk_clean.sample(
                    n=remaining,
                    random_state=RANDOM_SEED
                )

            chunks_kept.append(chunk_clean)

            rows_so_far += len(chunk_clean)

            if rows_so_far >= ROWS_PER_FILE:
                break

        if chunks_kept:

            dfs.append(
                pd.concat(
                    chunks_kept,
                    ignore_index=True
                )
            )

            print(f"✓ {fname} ({rows_so_far:,} rows)")

        else:
            print(f"✗ {fname} — no usable data")

    except Exception as e:
        print(f"✗ {fname} — skipped ({e})")


# combine all files
data = pd.concat(dfs, ignore_index=True)

print("\nDataset Loaded")
print("Total Rows:", len(data))
print("Total Columns:", len(data.columns))


# ============================================================
# SECTION 4 — LABEL PROCESSING
# ============================================================

print("\n[STEP 2] Processing labels...")

label_col = (
    "Label"
    if "Label" in data.columns
    else data.columns[-1]
)

# binary labels
data["is_attack"] = (
    data[label_col].str.strip() != "BENIGN"
).astype(int)

labels = data["is_attack"].values

# numeric features only
numeric_data = data.drop(
    columns=[label_col, "is_attack"],
    errors="ignore"
)

numeric_data = numeric_data.select_dtypes(
    include=[np.number]
)

# clean invalid values
numeric_data.replace(
    [np.inf, -np.inf],
    0,
    inplace=True
)

numeric_data.fillna(0, inplace=True)

print("Label Distribution:")
print(np.bincount(labels))


# ============================================================
# SECTION 5 — FEATURE SELECTION + SCALING
# ============================================================

print("\n[STEP 3] Feature selection and scaling...")

sample_size = min(5000, len(numeric_data))

idx = np.random.choice(
    len(numeric_data),
    sample_size,
    replace=False
)

X_sample = numeric_data.iloc[idx]
y_sample = labels[idx]

# Random Forest feature importance
rf = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_sample, y_sample)

importances = rf.feature_importances_

top_idx = np.argsort(importances)[::-1][:N_QUBITS]

selected_features = numeric_data.columns[top_idx].tolist()

print("\nSelected Features:")

for f in selected_features:
    print("-", f)

# scale features
X_selected = numeric_data[selected_features].copy()

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X_selected)

print("\nScaled Shape:", X_scaled.shape)


# ============================================================
# SECTION 6 — CREATE BALANCED TEST SET
# ============================================================

print("\n[STEP 4] Creating balanced dataset...")

normal_idx = np.where(labels == 0)[0]
attack_idx = np.where(labels == 1)[0]

test_normal = np.random.choice(
    normal_idx,
    50,
    replace=False
)

test_attack = np.random.choice(
    attack_idx,
    50,
    replace=False
)

test_idx = np.concatenate([
    test_normal,
    test_attack
])

np.random.shuffle(test_idx)

true_labels = labels[test_idx]

print("Balanced Distribution:")
print(np.bincount(true_labels))


# ============================================================
# SECTION 7 — QUANTUM CIRCUIT SIMULATION
# ============================================================

print("\n[STEP 5] Running quantum simulation...")

sim = AerSimulator()

SHOTS = 512

n_states = 2 ** N_QUBITS


# -------------------------
# Build Quantum Circuit
# -------------------------

def build_circuit(x):

    qc = QuantumCircuit(N_QUBITS)

    for i in range(N_QUBITS):
        qc.ry(np.pi * float(x[i]), i)

    # entanglement
    for i in range(N_QUBITS - 1):
        qc.cx(i, i + 1)

    qc.measure_all()

    return qc


# -------------------------
# Quantum Probabilities
# -------------------------

def get_probabilities(x):

    qc = transpile(
        build_circuit(x),
        sim
    )

    result = sim.run(
        qc,
        shots=SHOTS
    ).result()

    counts = result.get_counts()

    probs = np.zeros(n_states)

    for bitstring, count in counts.items():

        idx = int(bitstring, 2) % n_states

        probs[idx] = count / SHOTS

    return probs


# ============================================================
# SECTION 8 — BUILD QUANTUM BASELINE
# ============================================================

print("\n[STEP 6] Building quantum baseline...")

baseline = np.zeros(n_states)

for i in normal_idx[:30]:

    baseline += get_probabilities(
        X_scaled[i]
    )

baseline /= 30

baseline = np.clip(
    baseline,
    1e-10,
    1
)


# ============================================================
# SECTION 9 — QUANTUM ANOMALY DETECTION
# ============================================================

print("\n[STEP 7] Computing anomaly scores...")

distances = []

for i in tqdm(test_idx):

    sample_probs = get_probabilities(
        X_scaled[i]
    )

    sample_probs = np.clip(
        sample_probs,
        1e-10,
        1
    )

    dist = jensenshannon(
        baseline,
        sample_probs
    )

    distances.append(dist)

distances = np.array(distances)

print("Quantum scoring complete")


# ============================================================
# SECTION 10 — BUILD HYBRID DATASET
# ============================================================

print("\n[STEP 8] Building hybrid dataset...")

X_classical = X_scaled[test_idx]

quantum_feature = distances.reshape(-1, 1)

X_hybrid = np.hstack([
    X_classical,
    quantum_feature
])

print("Hybrid Shape:", X_hybrid.shape)


# ============================================================
# SECTION 11 — CLASSICAL IDS MODEL
# ============================================================

print("\n[STEP 9] Training classical IDS...")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_classical,
    true_labels,
    test_size=0.3,
    random_state=42
)

clf_classical = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

clf_classical.fit(
    X_train_c,
    y_train_c
)

pred_c = clf_classical.predict(X_test_c)

print("\nCLASSICAL RESULTS")
print(confusion_matrix(y_test_c, pred_c))

print(classification_report(
    y_test_c,
    pred_c,
    zero_division=0
))


# ============================================================
# SECTION 12 — QUANTUM HYBRID IDS MODEL
# ============================================================

print("\n[STEP 10] Training quantum hybrid IDS...")

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
    X_hybrid,
    true_labels,
    test_size=0.3,
    random_state=42
)

clf_quantum = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

clf_quantum.fit(
    X_train_q,
    y_train_q
)

pred_q = clf_quantum.predict(X_test_q)

print("\nQUANTUM HYBRID RESULTS")
print(confusion_matrix(y_test_q, pred_q))

print(classification_report(
    y_test_q,
    pred_q,
    zero_division=0
))


# ============================================================
# SECTION 13 — PERFORMANCE COMPARISON
# ============================================================

print("\n[STEP 11] Comparing models...")

classical_acc = accuracy_score(
    y_test_c,
    pred_c
)

quantum_acc = accuracy_score(
    y_test_q,
    pred_q
)

classical_auc = roc_auc_score(
    y_test_c,
    clf_classical.predict_proba(X_test_c)[:, 1]
)

quantum_auc = roc_auc_score(
    y_test_q,
    clf_quantum.predict_proba(X_test_q)[:, 1]
)

print("\nFINAL RESULTS")
print("--------------------------")
print("Classical Accuracy:", classical_acc)
print("Quantum Accuracy:", quantum_acc)

print("\nClassical AUC:", classical_auc)
print("Quantum AUC:", quantum_auc)


# ============================================================
# SECTION 14 — ROC CURVE VISUALIZATION
# ============================================================

print("\n[STEP 12] Generating ROC curve...")

y_prob_classical = clf_classical.predict_proba(
    X_test_c
)[:, 1]

y_prob_quantum = clf_quantum.predict_proba(
    X_test_q
)[:, 1]

fpr_c, tpr_c, _ = roc_curve(
    y_test_c,
    y_prob_classical
)

fpr_q, tpr_q, _ = roc_curve(
    y_test_q,
    y_prob_quantum
)

auc_c = auc(fpr_c, tpr_c)
auc_q = auc(fpr_q, tpr_q)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_c,
    tpr_c,
    linewidth=2,
    label=f"Classical IDS (AUC = {auc_c:.3f})"
)

plt.plot(
    fpr_q,
    tpr_q,
    linewidth=2,
    linestyle="--",
    label=f"Quantum Hybrid IDS (AUC = {auc_q:.3f})"
)

# random baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve — Classical vs Quantum Hybrid IDS"
)

plt.legend()

plt.tight_layout()

# save plot
plt.savefig(
    "roc_quantum_vs_classical.png",
    dpi=300
)

plt.show()

print("\nROC plot saved successfully")


# ============================================================
# SECTION 15 — FINAL SUMMARY
# ============================================================

print("\n" + "="*60)
print(" FINAL PROJECT SUMMARY ")
print("="*60)

print("\nProject:")
print("Quantum Hybrid Intrusion Detection System")

print("\nDataset:")
print("CIC-IDS-2017")

print("\nTechniques Used:")
print("- Classical Machine Learning")
print("- Quantum Circuit Simulation")
print("- Random Forest")
print("- Jensen-Shannon Distance")
print("- Hybrid Quantum-Classical Detection")

print("\nOutput:")
print("- Intrusion Detection")
print("- ROC Curve")
print("- Classical vs Quantum Comparison")

print("\nDone.")