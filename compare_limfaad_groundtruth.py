"""
Ground-truth comparison of XGBoost vs BERT vs KNN on the same LIMFAAD test set.
All 3 models were evaluated on the same 3,000 held-out samples (same split).
This script merges their predictions and produces:
  - Per-model metrics table
  - Agreement analysis (where all 3 agree / only 2 agree / all 3 disagree)
  - Per-class accuracy for each model
  - Head-to-head visualisations with ground truth
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

OUTPUT_DIR = "task3_limfaad/outputs"
CLASS_NAMES = ['Bot', 'Real', 'Scam', 'Spam']
COLORS = {'XGBoost': '#3498db', 'BERT': '#e67e22', 'KNN': '#2ecc71'}


def load_val(name, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)[['true_label', 'predicted_label', 'confidence']]
    df.columns = ['true_label', f'pred_{name}', f'conf_{name}']
    return df


def main():
    print("=" * 70)
    print("GROUND-TRUTH COMPARISON: XGBoost vs BERT vs KNN on LIMFAAD test set")
    print("=" * 70)

    # ── Load all 3 validation result CSVs ─────────────────────────────────
    xgb  = load_val('XGBoost', 'limfaad_model_validation_results.csv')
    bert = load_val('BERT',    'limfaad_bert_validation_results.csv')
    knn  = load_val('KNN',     'limfaad_knn_validation_results.csv')

    available = {k: v for k, v in {'XGBoost': xgb, 'BERT': bert, 'KNN': knn}.items() if v is not None}
    if not available:
        print("No validation files found. Run all three training scripts first.")
        return

    # Merge on index (same test set order)
    base = list(available.values())[0][['true_label']].copy()
    for name, df in available.items():
        base[f'pred_{name}'] = df[f'pred_{name}'].values
        base[f'conf_{name}'] = df[f'conf_{name}'].values
    df = base.reset_index(drop=True)
    models = list(available.keys())
    print(f"\nTest samples: {len(df)}, Models: {models}")

    # ── Per-model metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-MODEL METRICS (ground truth)")
    print("=" * 70)
    metrics = {}
    for name in models:
        y_true = df['true_label']
        y_pred = df[f'pred_{name}']
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_NAMES, average=None
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_NAMES, average='macro'
        )
        metrics[name] = {
            'accuracy': acc, 'macro_p': macro_p,
            'macro_r': macro_r, 'macro_f1': macro_f1,
            'precision': p, 'recall': r, 'f1': f1,
        }
        print(f"\n{name} — Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}")
        print(classification_report(y_true, y_pred, labels=CLASS_NAMES, target_names=CLASS_NAMES))

    # ── Agreement analysis ─────────────────────────────────────────────────
    print("=" * 70)
    print("AGREEMENT ANALYSIS")
    print("=" * 70)
    pred_cols = [f'pred_{m}' for m in models]

    if len(models) == 3:
        all_agree  = (df[pred_cols[0]] == df[pred_cols[1]]) & (df[pred_cols[1]] == df[pred_cols[2]])
        none_agree = (df[pred_cols[0]] != df[pred_cols[1]]) & \
                     (df[pred_cols[1]] != df[pred_cols[2]]) & \
                     (df[pred_cols[0]] != df[pred_cols[2]])
        two_agree  = ~all_agree & ~none_agree

        # All 3 agree + correct
        all_agree_correct = all_agree & (df[pred_cols[0]] == df['true_label'])

        print(f"\nAll 3 agree          : {all_agree.sum():>5} / {len(df)}  ({100*all_agree.mean():.1f}%)")
        print(f"  → of which correct : {all_agree_correct.sum():>5} ({100*all_agree_correct.sum()/all_agree.sum():.1f}% of agreements)")
        print(f"Exactly 2 agree      : {two_agree.sum():>5} / {len(df)}  ({100*two_agree.mean():.1f}%)")
        print(f"All 3 disagree       : {none_agree.sum():>5} / {len(df)}  ({100*none_agree.mean():.1f}%)")

        # Per-pair agreement
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                m1, m2 = models[i], models[j]
                agree = (df[f'pred_{m1}'] == df[f'pred_{m2}']).mean()
                print(f"{m1} vs {m2} agreement  : {agree:.2%}")

    # ── Save report ────────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, 'limfaad_groundtruth_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GROUND-TRUTH COMPARISON: XGBoost vs BERT vs KNN\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test samples: {len(df)}\n\n")
        f.write(f"{'Metric':<20} " + "  ".join(f"{m:<12}" for m in models) + "\n")
        f.write("-" * 60 + "\n")
        for key, label in [('accuracy','Accuracy'), ('macro_p','Macro Precision'),
                            ('macro_r','Macro Recall'), ('macro_f1','Macro F1')]:
            f.write(f"{label:<20} " + "  ".join(f"{metrics[m][key]:<12.4f}" for m in models) + "\n")
        if len(models) == 3:
            f.write(f"\nAll 3 agree: {all_agree.sum()} ({100*all_agree.mean():.1f}%)\n")
            f.write(f"All 3 agree & correct: {all_agree_correct.sum()} ({100*all_agree_correct.sum()/all_agree.sum():.1f}% of agreements)\n")
            f.write(f"All 3 disagree: {none_agree.sum()} ({100*none_agree.mean():.1f}%)\n")
    print(f"\nSaved report: {report_path}")

    # ── Visualisation 1: Per-model per-class accuracy bar chart ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall metrics grouped bar
    metric_labels = ['Accuracy', 'Macro P', 'Macro R', 'Macro F1']
    metric_keys   = ['accuracy', 'macro_p', 'macro_r', 'macro_f1']
    x = np.arange(len(metric_labels))
    w = 0.25
    for i, name in enumerate(models):
        vals = [metrics[name][k] for k in metric_keys]
        offset = (i - (len(models)-1)/2) * w
        bars = axes[0].bar(x + offset, vals, w, label=name,
                           color=COLORS.get(name, f'C{i}'), edgecolor='black', alpha=0.85)
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                         f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0.9, 1.02)
    axes[0].set_title('Overall Metrics (ground truth)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    # Per-class F1
    x2 = np.arange(len(CLASS_NAMES))
    for i, name in enumerate(models):
        offset = (i - (len(models)-1)/2) * w
        axes[1].bar(x2 + offset, metrics[name]['f1'], w, label=name,
                    color=COLORS.get(name, f'C{i}'), edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylim(0.85, 1.02)
    axes[1].set_title('Per-Class F1 Score (ground truth)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('XGBoost vs BERT vs KNN – LIMFAAD Test Set (Ground Truth)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    chart1 = os.path.join(OUTPUT_DIR, 'limfaad_groundtruth_metrics.png')
    plt.savefig(chart1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart1}")

    # ── Visualisation 2: Agreement pie + per-class agreement bar ──────────
    if len(models) == 3:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Pie: all agree correct / all agree wrong / 2 agree / all disagree
        all_agree_wrong = all_agree & (df[pred_cols[0]] != df['true_label'])
        sizes = [
            int(all_agree_correct.sum()),
            int(all_agree_wrong.sum()),
            int(two_agree.sum()),
            int(none_agree.sum()),
        ]
        labels = [
            f'All 3 agree\n& correct\n({sizes[0]})',
            f'All 3 agree\n& wrong\n({sizes[1]})',
            f'2 models agree\n({sizes[2]})',
            f'All 3 disagree\n({sizes[3]})',
        ]
        pie_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        axes[0].pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 9})
        axes[0].set_title('Model Agreement on 3,000 Test Samples', fontweight='bold')

        # Per-class: how often do all 3 agree AND correct
        class_agree_correct = {}
        for c in CLASS_NAMES:
            mask = df['true_label'] == c
            sub = df[mask]
            ag = (sub[pred_cols[0]] == sub[pred_cols[1]]) & (sub[pred_cols[1]] == sub[pred_cols[2]])
            correct = ag & (sub[pred_cols[0]] == sub['true_label'])
            class_agree_correct[c] = correct.sum() / len(sub)

        axes[1].bar(CLASS_NAMES, [class_agree_correct[c] for c in CLASS_NAMES],
                    color=['#3498db','#e74c3c','#2ecc71','#f39c12'], edgecolor='black')
        for i, (c, v) in enumerate(class_agree_correct.items()):
            axes[1].text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title('All 3 Models Agree & Correct — Per Class', fontweight='bold')
        axes[1].set_ylabel('Proportion of class samples')
        axes[1].grid(axis='y', linestyle='--', alpha=0.4)

        plt.suptitle('Model Agreement Analysis – LIMFAAD Test Set',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        chart2 = os.path.join(OUTPUT_DIR, 'limfaad_groundtruth_agreement.png')
        plt.savefig(chart2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {chart2}")

    print("\n" + "=" * 70)
    print("DONE — Ground-truth comparison complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
