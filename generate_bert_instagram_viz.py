"""
Generates BERT-specific figures for the Instagram negative-comment accounts:
  1. task4_classification/instagram_bert_category_distribution.png
  2. task4_classification/instagram_bert_confidence_distribution.png
"""

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH  = "task4_classification/instagram_negative_xgboost_vs_bert_vs_knn_results.csv"
OUT_DIR   = "task4_classification"

# ---------------------------------------------------------------------------
# Load BERT predictions
# ---------------------------------------------------------------------------
rows = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        rows.append(row)

labels      = [r['Predicted_Label_BERT'] for r in rows]
confidences = [float(r['Confidence_BERT']) for r in rows]

from collections import Counter
counts = Counter(labels)
classes     = ['Bot', 'Real', 'Scam', 'Spam']
class_counts = [counts.get(c, 0) for c in classes]
total        = sum(class_counts)

# ---------------------------------------------------------------------------
# Figure 1 – Category distribution
# ---------------------------------------------------------------------------
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(classes, class_counts, color=colors, edgecolor='black', linewidth=0.7, width=0.55)
ax.set_title('Negative Comments – Account Classification Distribution\n(BERT Model)', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Class', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_ylim(0, max(class_counts) * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, cnt in zip(bars, class_counts):
    pct = cnt / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f'{cnt}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

cat_path = os.path.join(OUT_DIR, 'instagram_bert_category_distribution.png')
plt.tight_layout()
plt.savefig(cat_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {cat_path}")

# ---------------------------------------------------------------------------
# Figure 2 – Confidence distribution
# ---------------------------------------------------------------------------
mean_conf = sum(confidences) / len(confidences)

fig, ax = plt.subplots(figsize=(9, 6))
n, bins, patches = ax.hist(confidences, bins=20, color='#3498db',
                           edgecolor='white', linewidth=0.6, alpha=0.85)
ax.axvline(mean_conf, color='red', linestyle='--', linewidth=1.8,
           label=f'Mean: {mean_conf:.3f}')
ax.set_title('Negative Comments – Confidence Distribution\n(BERT Model)', fontsize=13, fontweight='bold')
ax.set_xlabel('Confidence (Max Class Probability)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

conf_path = os.path.join(OUT_DIR, 'instagram_bert_confidence_distribution.png')
plt.tight_layout()
plt.savefig(conf_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {conf_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nBERT Instagram Predictions:")
for cls, cnt in zip(classes, class_counts):
    print(f"  {cls}: {cnt} ({cnt/total*100:.1f}%)")
print(f"\nConfidence — Mean: {mean_conf:.4f}  Min: {min(confidences):.4f}  Max: {max(confidences):.4f}")
