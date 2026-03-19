"""
Task 4: Classify Negative Comment User Profiles with Trained XGBoost Model
Uses LIMFAAD-trained XGBoost to classify accounts that posted negative comments.

Input: instagram_roberta_negative_comments.csv (55 accounts with negative comments)
Model: Trained XGBoost from Task 3 (LIMFAAD)
Output: Bot/Scam/Real/Spam classification + Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'input_file': 'task2_roberta/instagram_roberta_negative_comments.csv',
    'output_dir': 'task4_classification',
    'models_dir': 'task3_limfaad/models',
}
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("="*80)
print("TASK 4: CLASSIFY NEGATIVE-COMMENT USER PROFILES WITH XGBOOST")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Model: XGBoost (trained on LIMFAAD)")
print("Input: Negative comments user profiles (from Task 2)")
print("="*80)

# ---- Helper functions ----
def parse_count(x):
    """Parse Followers/Following like 136k, 1.2M, 11.6k -> int."""
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(',', '').lower()
    if s in ('', 'nan', 'no', 'yes'): return np.nan
    try:
        if s.endswith('k'):
            return int(float(s[:-1]) * 1000)
        if s.endswith('m'):
            return int(float(s[:-1]) * 1_000_000)
        return int(float(s))
    except Exception:
        return np.nan

def parse_mutual_friends(x):
    """Mutual_Friends: no->0, yes->1, else int."""
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s == 'no': return 0
    if s == 'yes': return 1
    try:
        return int(float(s))
    except Exception:
        return 0

# ---- Step 1: Load negative comments with user details ----
print("\n" + "="*80)
print("STEP 1: LOADING NEGATIVE COMMENTS USER PROFILES")
print("="*80)

df = pd.read_csv(CONFIG['input_file'])
print(f"Loaded {len(df)} accounts (users who posted negative comments)")
print(f"Columns: {list(df.columns)}")

has_labels = 'Labels' in df.columns
demo_metrics = False
if has_labels:
    valid = df['Labels'].isin(['Bot', 'Scam', 'Real', 'Spam'])
    has_labels = valid.sum() == len(df)
    if not has_labels:
        print(f"  Labels column present but {len(df) - valid.sum()} rows invalid; metrics will use demo mode.")
        has_labels = False
if has_labels:
    print("  Ground-truth 'Labels' found. Will compute real Accuracy, Precision, Recall, F1, Confusion Matrix.")
else:
    print("  No 'Labels' column. Will use predictions as pseudo-labels for metrics (demo only).")
    print("  Add 'Labels' (Bot/Scam/Real/Spam) per row for real evaluation.")
    demo_metrics = True

# ---- Step 2: Preprocess user profiles to match LIMFAAD format ----
print("\n" + "="*80)
print("STEP 2: PREPROCESSING USER PROFILES (match LIMFAAD format)")
print("="*80)

# Parse numeric columns
df['Followers'] = df['Followers'].apply(parse_count)
df['Following'] = df['Following'].apply(parse_count)
if 'Posts' not in df.columns:
    df['Posts'] = 0
else:
    df['Posts'] = pd.to_numeric(df['Posts'], errors='coerce').fillna(0).astype(int)

df['Followers'] = df['Followers'].fillna(0).astype(int)
df['Following'] = df['Following'].fillna(0).astype(int)

# Compute ratios (LIMFAAD uses Following/Followers, Posts/Followers)
ff = np.where(
    df['Followers'].astype(float) > 0,
    df['Following'].astype(float) / np.maximum(df['Followers'].astype(float), 1e-9),
    np.nan
)
df['Following/Followers'] = ff

pf = np.where(
    df['Followers'].astype(float) > 0,
    df['Posts'].astype(float) / np.maximum(df['Followers'].astype(float), 1e-9),
    0.0
)
df['Posts/Followers'] = pf

# Handle NaN ratios
df['Following/Followers'] = pd.to_numeric(df['Following/Followers'], errors='coerce')
df['Posts/Followers'] = pd.to_numeric(df['Posts/Followers'], errors='coerce')
ff_med = df['Following/Followers'].median()
pf_med = df['Posts/Followers'].median()
if np.isnan(ff_med): ff_med = 1.0
if np.isnan(pf_med): pf_med = 0.0
df['Following/Followers'] = df['Following/Followers'].fillna(ff_med)
df['Posts/Followers'] = df['Posts/Followers'].fillna(pf_med)

# Encode categoricals (match LIMFAAD)
for col, insta_col in [
    ('Bio', 'Bio'),
    ('Profile Picture', 'Profile_Picture'),
    ('External Link', 'External_Link'),
    ('Threads', 'Threads'),
]:
    if insta_col not in df.columns:
        df[col] = 0
        continue
    df[col] = df[insta_col].apply(
        lambda x: 1 if str(x).strip().lower() in ('yes', 'y') else 0
    )

# Mutual Friends
if 'Mutual_Friends' in df.columns:
    df['Mutual Friends'] = df['Mutual_Friends'].apply(parse_mutual_friends)
else:
    df['Mutual Friends'] = 0

print("✅ Preprocessing complete")

# ---- Step 3: Load trained XGBoost model ----
print("\n" + "="*80)
print("STEP 3: LOADING TRAINED XGBOOST MODEL")
print("="*80)

model_path = os.path.join(CONFIG['models_dir'], 'limfaad_xgboost_model.json')
encoder_path = os.path.join(CONFIG['models_dir'], 'limfaad_label_encoder.pkl')
info_path = os.path.join(CONFIG['models_dir'], 'limfaad_feature_info.pkl')

model = xgb.XGBClassifier()
model.load_model(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
with open(info_path, 'rb') as f:
    feature_info = pickle.load(f)

feature_columns = feature_info['feature_columns']
class_names = np.array(feature_info['class_names'])
print(f"Model loaded from {model_path}")
print(f"Classes: {class_names}")

# ---- Step 4: Run inference ----
print("\n" + "="*80)
print("STEP 4: RUNNING INFERENCE")
print("="*80)

X = df[feature_columns].copy()
X = X[feature_columns]  # Ensure order
X = X.fillna(0)

y_pred = model.predict(X)
y_proba = model.predict_proba(X)
pred_labels = label_encoder.inverse_transform(y_pred)

df['Predicted_Label'] = pred_labels
df['Confidence'] = [float(np.max(p)) for p in y_proba]
for i, c in enumerate(class_names):
    df[f'Prob_{c}'] = y_proba[:, i]

print(f"Predicted {len(df)} accounts.")
dist = df['Predicted_Label'].value_counts().reindex(class_names, fill_value=0)
for c in class_names:
    print(f"  {c}: {dist.get(c, 0)}")

# ---- Step 5: Metrics (if Labels available) ----
if has_labels or demo_metrics:
    y_true = df['Labels'].values if has_labels else pred_labels
    acc = accuracy_score(y_true, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred_labels, labels=class_names, average=None
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, pred_labels, labels=class_names, average='macro'
    )
    cm = confusion_matrix(y_true, pred_labels, labels=class_names)

    print("\n" + "="*80)
    print("METRICS" + (" (ground-truth Labels)" if has_labels else " (demo: pseudo-labels)"))
    print("="*80)
    print(f"Accuracy: {acc:.4f} ({100*acc:.2f}%)")
    print(f"Macro Precision: {macro_p:.4f}  Macro Recall: {macro_r:.4f}  Macro F1: {macro_f1:.4f}")
    print("Per-class:")
    for i, c in enumerate(class_names):
        print(f"  {c}: P={precision[i]:.4f} R={recall[i]:.4f} F1={f1[i]:.4f} support={int(support[i])}")

# ---- Step 6: Save results ----
print("\n" + "="*80)
print("STEP 6: SAVING RESULTS")
print("="*80)

base_cols = ['Post_ID', 'Post_Text', 'User_Name', 'Followers', 'Following', 'Posts',
             'Predicted_Label', 'Confidence'] + [f'Prob_{c}' for c in class_names] + ['Labels']
out_cols = [c for c in base_cols if c in df.columns]
results = df[out_cols].copy()
results_file = os.path.join(CONFIG['output_dir'], 'instagram_negative_classification_results.csv')
results.to_csv(results_file, index=False)
print(f"Saved: {results_file}")

# Report
report_file = os.path.join(CONFIG['output_dir'], 'instagram_negative_classification_report.txt')
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("TASK 4: NEGATIVE-COMMENT USER PROFILES CLASSIFICATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: XGBoost (LIMFAAD-trained)\n")
    f.write(f"Input: {CONFIG['input_file']}\n")
    f.write(f"Total accounts: {len(df)}\n\n")
    f.write("Classification distribution:\n")
    for c in class_names:
        count = dist.get(c, 0)
        f.write(f"  {c}: {count} ({100*count/len(df):.1f}%)\n")
    if has_labels or demo_metrics:
        h = "METRICS (ground-truth Labels)" if has_labels else "METRICS (demo: pseudo-labels)"
        f.write("\n" + "="*80 + "\n" + h + "\n" + "="*80 + "\n\n")
        f.write(f"Accuracy: {acc:.4f} ({100*acc:.2f}%)\n")
        f.write(f"Macro Precision: {macro_p:.4f}\n")
        f.write(f"Macro Recall: {macro_r:.4f}\n")
        f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
        f.write("Per-class:\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*60 + "\n")
        for i, c in enumerate(class_names):
            f.write(f"{c:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {int(support[i]):<10}\n")
        f.write("\n")
        f.write(classification_report(y_true, pred_labels, target_names=class_names))
        if demo_metrics:
            f.write("\n(Demo metrics use predictions as pseudo-labels. Add 'Labels' for real evaluation.)\n")
print(f"Saved: {report_file}")

# ---- Step 7: Visualizations ----
print("\n" + "="*80)
print("STEP 7: GENERATING VISUALIZATIONS")
print("="*80)

# Confusion matrix (if metrics available)
if has_labels or demo_metrics:
    cm_title = 'Negative Comments – Account Classification Confusion Matrix' + (' [demo]' if demo_metrics else '')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title(cm_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(CONFIG['output_dir'], 'instagram_negative_classification_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # Metrics visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                [acc, macro_p, macro_r, macro_f1],
                color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], edgecolor='black')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Overall Metrics' + (' [demo]' if demo_metrics else ''))
    for i, v in enumerate([acc, macro_p, macro_r, macro_f1]):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

    x = np.arange(len(class_names))
    w = 0.25
    axes[1].bar(x - w, precision, w, label='Precision', color='#3498db')
    axes[1].bar(x, recall, w, label='Recall', color='#e74c3c')
    axes[1].bar(x + w, f1, w, label='F1-Score', color='#f39c12')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Per-Class Metrics')
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    metrics_path = os.path.join(CONFIG['output_dir'], 'instagram_negative_metrics_visualization.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {metrics_path}")

# Category distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(dist.index, dist.values, color=['#3498db','#e74c3c','#2ecc71','#f39c12'], edgecolor='black')
ax.set_xlabel('Predicted Class')
ax.set_ylabel('Count')
ax.set_title('Negative Comments – Account Classification Distribution')
for i, (k, v) in enumerate(dist.items()):
    ax.text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')
plt.tight_layout()
dist_path = os.path.join(CONFIG['output_dir'], 'instagram_negative_category_distribution.png')
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {dist_path}")

# Confidence distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['Confidence'], bins=15, color='steelblue', edgecolor='black', alpha=0.8)
ax.axvline(df['Confidence'].mean(), color='red', linestyle='--', label=f"Mean: {df['Confidence'].mean():.3f}")
ax.set_xlabel('Confidence')
ax.set_ylabel('Frequency')
ax.set_title('Negative Comments – Confidence Distribution')
ax.legend()
plt.tight_layout()
conf_path = os.path.join(CONFIG['output_dir'], 'instagram_negative_confidence_distribution.png')
plt.savefig(conf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {conf_path}")

print("\n" + "="*80)
print("TASK 4 COMPLETE")
print("="*80)
print(f"\nOutputs: {results_file}, {report_file}, {dist_path}, {conf_path}")
if has_labels or demo_metrics:
    print(f"        {cm_path}, {metrics_path}")
print("\n📊 Research Question: Correlation between negative comments and spam/scam/bot accounts")
print("="*80)
