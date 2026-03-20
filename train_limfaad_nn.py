"""
Task 3 (Neural Network variant): Train MLP Neural Network on LIMFAAD for account classification.
Uses the same LIMFAAD data and train/test split as XGBoost and KNN, but trains a deep
Multi-Layer Perceptron (MLP) for 4-class classification (Bot, Scam, Real, Spam).
Enables direct comparison: XGBoost (tree-based) vs Neural Network (deep learning) vs KNN.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config (aligned with other training scripts for same split)
# ---------------------------------------------------------------------------
CONFIG = {
    'input_file': 'LIMFADD.csv',
    'output_dir': 'task3_limfaad/outputs',
    'models_dir': 'task3_limfaad/models',
    'test_size': 0.2,
    'random_state': 42,
    'nn_params': {
        'hidden_sizes': [256, 128, 64],
        'dropout_rate': 0.3,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 150,
        'patience': 20,
    }
}

FEATURE_COLUMNS = [
    'Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers',
    'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads'
]

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)

print("=" * 80)
print("TASK 3: TRAIN NEURAL NETWORK (MLP) MODEL ON LIMFAAD DATASET")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Model: Multi-Layer Perceptron (MLP) Neural Network")
print("Dataset: LIMFADD.csv")
print("=" * 80)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')

DEVICE = get_device()
print(f"\nUsing device: {DEVICE}")


# ---------------------------------------------------------------------------
# MLP Architecture
# ---------------------------------------------------------------------------
class AccountClassifierMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 1: LOADING LIMFAAD DATASET")
print("=" * 80)

df = pd.read_csv(CONFIG['input_file'])
print(f"Loaded {len(df)} samples, {len(df.columns) - 1} features")
print(f"\nClass distribution:")
print(df['Labels'].value_counts().sort_index())


# ---------------------------------------------------------------------------
# Step 2: Preprocessing (same as XGBoost/KNN scripts)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

df_processed = df.copy()
print("Keeping original dataset labels (Bot/Scam/Real/Spam)")

df_processed['Following/Followers'] = pd.to_numeric(
    df_processed['Following/Followers'], errors='coerce'
)
df_processed['Posts/Followers'] = pd.to_numeric(
    df_processed['Posts/Followers'], errors='coerce'
)

df_processed.loc[
    (df_processed['Following/Followers'].isna()) & (df_processed['Followers'] == 0),
    'Following/Followers'
] = df_processed['Following'].max()

df_processed.loc[
    (df_processed['Posts/Followers'].isna()) & (df_processed['Followers'] == 0),
    'Posts/Followers'
] = 0

df_processed['Following/Followers'].fillna(
    df_processed['Following/Followers'].median(), inplace=True
)
df_processed['Posts/Followers'].fillna(
    df_processed['Posts/Followers'].median(), inplace=True
)

for col in ['Bio', 'Profile Picture', 'External Link', 'Threads']:
    df_processed[col] = df_processed[col].apply(
        lambda x: 1 if str(x).strip().lower() in ('yes', 'y', '1') else 0
    )

print("Encoded categorical features (Bio, Profile Picture, External Link, Threads)")

X = df_processed[FEATURE_COLUMNS].copy()
y = df_processed['Labels'].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
num_classes = len(class_names)

print(f"\nFeatures: {FEATURE_COLUMNS}")
print(f"Classes: {list(class_names)}")


# ---------------------------------------------------------------------------
# Step 3: Train/Test Split
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 3: SPLITTING DATASET")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y_encoded
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")

# Further split train into train/val for early stopping
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=CONFIG['random_state'],
    stratify=y_train
)

print(f"Val set (for early stopping): {len(X_val)} samples")


# ---------------------------------------------------------------------------
# Step 4: Feature Scaling
# ---------------------------------------------------------------------------
# Ensure numeric and finite values before scaling
def clean_features(df_or_arr):
    arr = df_or_arr.values if hasattr(df_or_arr, 'values') else df_or_arr
    arr = arr.astype(float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    # Clip extreme outliers to 99.9th percentile range to prevent NaN gradients
    return arr

X_train_fit_clean = clean_features(X_train_fit)
X_val_clean       = clean_features(X_val)
X_test_clean      = clean_features(X_test)
X_train_clean     = clean_features(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fit_clean)
X_val_scaled   = scaler.transform(X_val_clean)
X_test_scaled  = scaler.transform(X_test_clean)
X_train_all_scaled = scaler.transform(X_train_clean)

# Safety clip after scaling to prevent extreme values
clip_val = 10.0
X_train_scaled     = np.clip(X_train_scaled,     -clip_val, clip_val)
X_val_scaled       = np.clip(X_val_scaled,       -clip_val, clip_val)
X_test_scaled      = np.clip(X_test_scaled,      -clip_val, clip_val)
X_train_all_scaled = np.clip(X_train_all_scaled, -clip_val, clip_val)

print(f"Features standardized (StandardScaler, clipped to ±{clip_val})")
print(f"  Train scaled: min={X_train_scaled.min():.2f}, max={X_train_scaled.max():.2f}, NaN={np.isnan(X_train_scaled).sum()}")


# ---------------------------------------------------------------------------
# Step 5: Build DataLoaders
# ---------------------------------------------------------------------------
def make_loader(X_np, y_np, batch_size, shuffle=True):
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

params = CONFIG['nn_params']
train_loader = make_loader(X_train_scaled, y_train_fit, params['batch_size'])
val_loader   = make_loader(X_val_scaled,   y_val,       params['batch_size'], shuffle=False)


# ---------------------------------------------------------------------------
# Step 6: Train
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4: TRAINING NEURAL NETWORK MODEL")
print("=" * 80)

model = AccountClassifierMLP(
    input_size=len(FEATURE_COLUMNS),
    hidden_sizes=params['hidden_sizes'],
    num_classes=num_classes,
    dropout_rate=params['dropout_rate'],
).to(DEVICE)

print(f"\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")

optimizer = optim.Adam(
    model.parameters(),
    lr=params['learning_rate'],
    weight_decay=params['weight_decay'],
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8
)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
patience_counter = 0
best_state = {k: v.clone() for k, v in model.state_dict().items()}
train_losses, val_losses, val_accs = [], [], []

print(f"\nTraining for up to {params['epochs']} epochs (early stopping patience={params['patience']})...")

for epoch in range(1, params['epochs'] + 1):
    # Train
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
    epoch_loss /= len(X_train_scaled)
    train_losses.append(epoch_loss)

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            val_loss += criterion(logits, y_batch).item() * len(X_batch)
            val_correct += (logits.argmax(1) == y_batch).sum().item()
    val_loss /= len(X_val_scaled)
    val_acc = val_correct / len(X_val_scaled)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= params['patience']:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {params['patience']} epochs)")
            break

# Load best weights
model.load_state_dict(best_state)
print("\nLoaded best model weights.")


# ---------------------------------------------------------------------------
# Step 7: Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 5: MODEL EVALUATION")
print("=" * 80)

def predict(X_np):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    pred = np.argmax(proba, axis=1)
    return pred, proba

y_train_pred, y_train_proba = predict(X_train_all_scaled)
y_test_pred,  y_test_proba  = predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy  = accuracy_score(y_test,  y_test_pred)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, average=None, labels=range(num_classes)
)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='macro'
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='weighted'
)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\nPer-Class Metrics (Test Set):")
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 60)
for i, cn in enumerate(class_names):
    print(f"{cn:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
print(f"\nMacro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1-Score:  {macro_f1:.4f}")


# ---------------------------------------------------------------------------
# Step 8: Save Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 6: SAVING RESULTS")
print("=" * 80)

# Training metrics CSV (same format as other models)
training_metrics = pd.DataFrame({
    'metric': ['train_accuracy', 'test_accuracy', 'macro_precision', 'macro_recall',
               'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1'],
    'value': [train_accuracy, test_accuracy, macro_precision, macro_recall,
              macro_f1, weighted_precision, weighted_recall, weighted_f1]
})
metrics_file = f"{CONFIG['output_dir']}/limfaad_nn_training_metrics.csv"
training_metrics.to_csv(metrics_file, index=False)
print(f"Saved: {metrics_file}")

# Validation results CSV (same format as other models)
validation_results = pd.DataFrame({
    'true_label':      [class_names[y] for y in y_test],
    'predicted_label': [class_names[y] for y in y_test_pred],
    'confidence':      [float(np.max(p)) for p in y_test_proba],
})
for i, cn in enumerate(class_names):
    validation_results[f'prob_{cn.lower()}'] = y_test_proba[:, i]
val_results_file = f"{CONFIG['output_dir']}/limfaad_nn_validation_results.csv"
validation_results.to_csv(val_results_file, index=False)
print(f"Saved: {val_results_file}")

# Text report (format matches generate_bert_metrics_viz.py regex patterns)
report_file = f"{CONFIG['output_dir']}/limfaad_nn_model_report.txt"
clf_report = classification_report(y_test, y_test_pred, target_names=class_names)
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("LIMFAAD NEURAL NETWORK MODEL EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model: Multi-Layer Perceptron (MLP) Neural Network\n")
    f.write(f"Architecture: {len(FEATURE_COLUMNS)} → {' → '.join(str(h) for h in params['hidden_sizes'])} → {num_classes}\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {CONFIG['input_file']}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")

    f.write("=" * 80 + "\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")

    f.write("=" * 80 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(clf_report)
print(f"Saved: {report_file}")


# ---------------------------------------------------------------------------
# Step 9: Save Model Artifacts
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 7: SAVING MODEL")
print("=" * 80)

# Save PyTorch model
model_file = f"{CONFIG['models_dir']}/limfaad_nn_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': len(FEATURE_COLUMNS),
        'hidden_sizes': params['hidden_sizes'],
        'num_classes': num_classes,
        'dropout_rate': params['dropout_rate'],
    }
}, model_file)
print(f"Saved: {model_file}")

# Save scaler
scaler_file = f"{CONFIG['models_dir']}/limfaad_nn_scaler.pkl"
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved: {scaler_file}")

# Save label encoder
encoder_file = f"{CONFIG['models_dir']}/limfaad_nn_label_encoder.pkl"
with open(encoder_file, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Saved: {encoder_file}")

# Save feature info
feature_info = {
    'feature_columns': FEATURE_COLUMNS,
    'class_names': class_names.tolist(),
}
feature_info_file = f"{CONFIG['models_dir']}/limfaad_nn_feature_info.pkl"
with open(feature_info_file, 'wb') as f:
    pickle.dump(feature_info, f)
print(f"Saved: {feature_info_file}")

# Save scaler mean/scale for fast NumPy inference (same as KNN)
np.save(f"{CONFIG['models_dir']}/limfaad_nn_scaler_mean.npy", scaler.mean_)
np.save(f"{CONFIG['models_dir']}/limfaad_nn_scaler_scale.npy", scaler.scale_)
print(f"Saved scaler mean/scale .npy arrays")


# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3 (NEURAL NETWORK) COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  Test Accuracy:   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Macro F1-Score:  {macro_f1:.4f}")
print(f"  Weighted F1:     {weighted_f1:.4f}")
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
