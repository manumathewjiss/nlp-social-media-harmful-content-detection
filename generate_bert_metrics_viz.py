"""
Generate a metrics visualization for the BERT model that matches the style
of the XGBoost limfaad_metrics_visualization.png.
Reads from limfaad_bert_model_report.txt.
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Parse BERT report
# ---------------------------------------------------------------------------
REPORT_PATH = "task3_limfaad/outputs/limfaad_bert_model_report.txt"
OUTPUT_PATH = "task3_limfaad/outputs/limfaad_bert_metrics_visualization.png"

report_text = open(REPORT_PATH).read()

# Overall metrics
accuracy  = float(re.search(r'Test Accuracy:\s+([\d.]+)', report_text).group(1)) * 100
macro_f1  = float(re.search(r'Macro F1:\s+([\d.]+)', report_text).group(1)) * 100

# Per-class lines  (Bot / Real / Scam / Spam)
classes = ['Bot', 'Real', 'Scam', 'Spam']
per_class = {}
for cls in classes:
    m = re.search(
        rf'\s+{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        report_text
    )
    per_class[cls] = {
        'precision': float(m.group(1)) * 100,
        'recall':    float(m.group(2)) * 100,
        'f1':        float(m.group(3)) * 100,
    }

# macro avg line for overall precision / recall
m_avg = re.search(r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_text)
macro_prec   = float(m_avg.group(1)) * 100
macro_recall = float(m_avg.group(2)) * 100

# ---------------------------------------------------------------------------
# Build figure (same 2×2 layout as XGBoost version)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "LIMFAAD BERT Model – Comprehensive Performance Metrics\n"
    "Bot / Scam / Real / Spam Classification",
    fontsize=14, fontweight='bold'
)

# ── Top-left: Overall metrics ────────────────────────────────────────────────
ax = axes[0, 0]
overall_labels  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
overall_values  = [accuracy, macro_prec, macro_recall, macro_f1]
overall_colors  = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax.bar(overall_labels, overall_values, color=overall_colors, edgecolor='black', linewidth=0.6)
ax.set_ylim(0, 110)
ax.set_title('Overall Model Performance Metrics', fontsize=11, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=10)
for bar, val in zip(bars, overall_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.tick_params(axis='x', labelsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Top-right: Per-class grouped bars ────────────────────────────────────────
ax = axes[0, 1]
x     = np.arange(len(classes))
width = 0.26
prec_vals    = [per_class[c]['precision'] for c in classes]
recall_vals  = [per_class[c]['recall']    for c in classes]
f1_vals      = [per_class[c]['f1']        for c in classes]

b1 = ax.bar(x - width, prec_vals,   width, label='Precision', color='#2ecc71', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x,         recall_vals, width, label='Recall',    color='#3498db', edgecolor='black', linewidth=0.5)
b3 = ax.bar(x + width, f1_vals,     width, label='F1-Score',  color='#e74c3c', edgecolor='black', linewidth=0.5)

ax.set_ylim(0, 115)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=9)
ax.set_title('Per-Class Performance Metrics', fontsize=11, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bars_group in [b1, b2, b3]:
    for bar in bars_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

# ── Bottom-left: Per-class Precision ────────────────────────────────────────
ax = axes[1, 0]
prec_sorted = sorted(zip(classes, prec_vals), key=lambda x: x[1])
cls_names_s = [c for c, _ in prec_sorted]
prec_s      = [v for _, v in prec_sorted]
bars = ax.barh(cls_names_s, prec_s, color='#3498db', edgecolor='black', linewidth=0.5)
ax.set_xlim(0, 115)
ax.set_title('Per-Class Precision', fontsize=11, fontweight='bold')
ax.set_xlabel('Precision (%)', fontsize=10)
for bar, val in zip(bars, prec_s):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', ha='left', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Bottom-right: Per-class F1-Score ─────────────────────────────────────────
ax = axes[1, 1]
f1_sorted  = sorted(zip(classes, f1_vals), key=lambda x: x[1])
cls_names_f = [c for c, _ in f1_sorted]
f1_s        = [v for _, v in f1_sorted]
bars = ax.barh(cls_names_f, f1_s, color='#f39c12', edgecolor='black', linewidth=0.5)
ax.set_xlim(0, 115)
ax.set_title('Per-Class F1-Score', fontsize=11, fontweight='bold')
ax.set_xlabel('F1-Score (%)', fontsize=10)
for bar, val in zip(bars, f1_s):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', ha='left', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUTPUT_PATH}")
