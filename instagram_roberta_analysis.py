"""
Task 2: RoBERTa Sentiment Analysis on Instagram 150 Comments
Extract negative comments from 150 entries. User details kept for Task 4 (XGBoost).

- Process ONLY comment text (Post_Text) with RoBERTa. No user details used for sentiment.
- Model: cardiffnlp/twitter-roberta-base-sentiment
- Output: Full sentiment results + negative comments file WITH user details (for Task 4).
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix (real if Sentiment column present, else demo).
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datetime import datetime
from preprocess import clean_text
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'input_file': 'task1_dataset/InstagramPosts_Base.csv',
    'output_dir': 'task2_roberta',
    'model_name': 'cardiffnlp/twitter-roberta-base-sentiment',
    'cache_dir': './model_cache',
    'batch_size': 16,
    'max_length': 128,
}
LABELS = ['negative', 'neutral', 'positive']

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'], exist_ok=True)

print("="*80)
print("TASK 2: ROBERTA SENTIMENT ANALYSIS ON INSTAGRAM 150 COMMENTS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: {CONFIG['model_name']}")
print(f"Input: {CONFIG['input_file']} (comments only for sentiment)")
print("="*80)

# ---- Step 1: Load Instagram dataset ----
print("\n" + "="*80)
print("STEP 1: LOADING INSTAGRAM DATASET")
print("="*80)

df = pd.read_csv(CONFIG['input_file'])
print(f"Loaded {len(df)} rows from {CONFIG['input_file']}")
print(f"Columns: {list(df.columns)}")

has_sentiment = 'Sentiment' in df.columns
demo_metrics = False
if has_sentiment:
    valid = df['Sentiment'].str.lower().isin(['negative', 'neutral', 'positive'])
    has_sentiment = valid.sum() == len(df)
    if not has_sentiment:
        print("  'Sentiment' column present but some values invalid; metrics will use demo mode.")
        has_sentiment = False
if has_sentiment:
    print("  Ground-truth 'Sentiment' found. Will compute Accuracy, Precision, Recall, F1, Confusion Matrix.")
else:
    print("  No 'Sentiment' column. Will use predictions as pseudo-labels for metrics (demo only).")
    print("  Add 'Sentiment' (negative/neutral/positive) per row for real evaluation.")
    demo_metrics = True

# ---- Step 2: Preprocess comments only ----
print("\n" + "="*80)
print("STEP 2: PREPROCESSING COMMENTS (Post_Text only)")
print("="*80)

df['clean_text'] = df['Post_Text'].fillna('').astype(str).apply(clean_text)
before = len(df)
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
dropped = before - len(df)
if dropped:
    print(f"Dropped {dropped} empty comments. {len(df)} remaining.")
else:
    print(f"All {len(df)} comments valid.")

# ---- Step 3: Load RoBERTa model ----
print("\n" + "="*80)
print("STEP 3: LOADING ROBERTA MODEL")
print("="*80)

print(f"Model: {CONFIG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG['model_name'], cache_dir=CONFIG['cache_dir'], local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'], cache_dir=CONFIG['cache_dir'], local_files_only=True
)
model.eval()
print("Model loaded.")

# ---- Step 4: Run sentiment on comments only ----
print("\n" + "="*80)
print("STEP 4: RUNNING SENTIMENT ANALYSIS (comments only)")
print("="*80)

texts = df['clean_text'].tolist()
predictions = []
confidences = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), CONFIG['batch_size']), desc="RoBERTa", unit="batch"):
        batch = texts[i:i + CONFIG['batch_size']]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG['max_length'],
        )
        out = model(**tokens)
        probs = F.softmax(out.logits, dim=1)
        for p in probs:
            idx = p.argmax().item()
            predictions.append(LABELS[idx])
            confidences.append(float(p[idx]))

df['Predicted_Sentiment'] = predictions
df['Confidence'] = confidences

print(f"\nProcessed {len(df)} comments.")
dist = df['Predicted_Sentiment'].value_counts()
for s in LABELS:
    c = dist.get(s, 0)
    print(f"  {s}: {c} ({100*c/len(df):.1f}%)")

# ---- Step 5: Extract negative comments (with user details) ----
print("\n" + "="*80)
print("STEP 5: EXTRACTING NEGATIVE COMMENTS (with user details for Task 4)")
print("="*80)

negative = df[df['Predicted_Sentiment'] == 'negative'].copy()
print(f"Negative comments: {len(negative)}")

# ---- Step 6: Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix) ----
y_pred = df['Predicted_Sentiment'].values
y_true = df['Sentiment'].str.lower().values if has_sentiment else y_pred

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=LABELS, average=None
)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=LABELS, average='macro'
)
cm = confusion_matrix(y_true, y_pred, labels=LABELS)

print("\n" + "="*80)
print("METRICS" + (" (ground-truth Sentiment)" if has_sentiment else " (demo: pseudo-labels)"))
print("="*80)
print(f"Accuracy: {acc:.4f} ({100*acc:.2f}%)")
print(f"Macro Precision: {macro_p:.4f}  Macro Recall: {macro_r:.4f}  Macro F1: {macro_f1:.4f}")
print("Per-class:")
for i, s in enumerate(LABELS):
    print(f"  {s}: P={precision[i]:.4f} R={recall[i]:.4f} F1={f1[i]:.4f} support={int(support[i])}")

# ---- Step 7: Save results ----
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

full_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_sentiment_results.csv')
df.to_csv(full_path, index=False)
print(f"Saved: {full_path}")

neg_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_negative_comments.csv')
negative.to_csv(neg_path, index=False)
print(f"Saved: {neg_path} (user details included for Task 4 XGBoost)")

report_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_metrics_report.txt')
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("TASK 2: INSTAGRAM ROBERTA SENTIMENT ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: {CONFIG['model_name']}\n")
    f.write(f"Input: {CONFIG['input_file']}\n")
    f.write(f"Total comments: {len(df)}\n")
    f.write(f"Negative comments: {len(negative)}\n\n")
    f.write("Predicted sentiment distribution:\n")
    for s in LABELS:
        c = dist.get(s, 0)
        f.write(f"  {s}: {c} ({100*c/len(df):.1f}%)\n")
    f.write("\n" + "="*80 + "\n")
    h = "METRICS (ground-truth Sentiment)" if has_sentiment else "METRICS (demo: pseudo-labels)"
    f.write(h + "\n")
    f.write("="*80 + "\n\n")
    f.write(f"Accuracy: {acc:.4f} ({100*acc:.2f}%)\n")
    f.write(f"Macro Precision: {macro_p:.4f}\n")
    f.write(f"Macro Recall: {macro_r:.4f}\n")
    f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
    f.write("Per-class:\n")
    f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
    f.write("-"*60 + "\n")
    for i, s in enumerate(LABELS):
        f.write(f"{s:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {int(support[i]):<10}\n")
    f.write("\n")
    f.write(classification_report(y_true, y_pred, target_names=LABELS))
    if demo_metrics:
        f.write("\n(Demo metrics use predictions as pseudo-labels. Add 'Sentiment' for real evaluation.)\n")
    f.write("\nNegative comments file includes user details for Task 4 XGBoost.\n")
print(f"Saved: {report_path}")

# Confusion matrix
cm_title = 'Instagram 150 – RoBERTa Confusion Matrix' + (' [demo: pseudo-labels]' if demo_metrics else '')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABELS, yticklabels=LABELS,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title(cm_title, fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
cm_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cm_path}")

# Metrics visualization (Accuracy, Precision, Recall, F1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            [acc, macro_p, macro_r, macro_f1],
            color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], edgecolor='black')
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('Score')
axes[0].set_title('Overall Metrics' + (' [demo]' if demo_metrics else ''))
for i, (name, v) in enumerate(zip(['Accuracy', 'Precision', 'Recall', 'F1'], [acc, macro_p, macro_r, macro_f1])):
    axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

x = np.arange(len(LABELS))
w = 0.25
axes[1].bar(x - w, precision, w, label='Precision', color='#3498db')
axes[1].bar(x, recall, w, label='Recall', color='#e74c3c')
axes[1].bar(x + w, f1, w, label='F1-Score', color='#f39c12')
axes[1].set_xticks(x)
axes[1].set_xticklabels(LABELS)
axes[1].set_ylabel('Score')
axes[1].set_title('Per-Class Metrics')
axes[1].legend()
axes[1].set_ylim(0, 1.05)
plt.tight_layout()
metrics_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_metrics_visualization.png')
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {metrics_path}")

# Sentiment distribution bar chart
fig, ax = plt.subplots(figsize=(8, 5))
counts = [dist.get(s, 0) for s in LABELS]
colors = ['#e74c3c', '#f39c12', '#2ecc71']
ax.bar(LABELS, counts, color=colors, edgecolor='black')
ax.set_ylabel('Count')
ax.set_title('RoBERTa Sentiment Distribution')
for i, v in enumerate(counts):
    ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
dist_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_sentiment_distribution.png')
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {dist_path}")

# Confidence distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['Confidence'], bins=20, color='steelblue', edgecolor='black', alpha=0.8)
ax.axvline(df['Confidence'].mean(), color='red', linestyle='--', label=f"Mean: {df['Confidence'].mean():.3f}")
ax.set_xlabel('Confidence')
ax.set_ylabel('Frequency')
ax.set_title('RoBERTa Confidence Distribution')
ax.legend()
plt.tight_layout()
conf_path = os.path.join(CONFIG['output_dir'], 'instagram_roberta_confidence_distribution.png')
plt.savefig(conf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {conf_path}")

print("\n" + "="*80)
print("TASK 2 COMPLETE")
print("="*80)
print(f"Outputs: {full_path}, {neg_path}, {report_path}")
print(f"        {cm_path}, {metrics_path}, {dist_path}, {conf_path}")
print(f"Next: Task 4 – run user profiles of negative comments through trained XGBoost.")
print(f"Use: {neg_path}")
print("="*80)
