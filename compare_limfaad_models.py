"""
Compare XGBoost vs Neural Network vs KNN on the LIMFAAD task (same test set).
Reads saved metrics CSVs from task3_limfaad/outputs and produces:
  - Console table
  - Combined bar chart (task3_limfaad/outputs/limfaad_model_comparison.png)
  - Side-by-side confusion matrices (limfaad_model_comparison_confusion_matrices.png)
  - Text report (limfaad_xgboost_vs_nn_vs_knn_comparison.txt)

Run after: train_limfaad_model.py, train_limfaad_nn.py, train_limfaad_knn.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

OUTPUT_DIR = "task3_limfaad/outputs"
MODELS_DIR = "task3_limfaad/models"

METRICS_FILES = {
    'XGBoost':        'limfaad_training_metrics.csv',
    'Neural Network': 'limfaad_nn_training_metrics.csv',
    'KNN':            'limfaad_knn_training_metrics.csv',
}

COLORS = {'XGBoost': '#3498db', 'Neural Network': '#e74c3c', 'KNN': '#2ecc71'}
REPORT_FILE = os.path.join(OUTPUT_DIR, 'limfaad_xgboost_vs_nn_vs_knn_comparison.txt')
CHART_FILE = os.path.join(OUTPUT_DIR, 'limfaad_model_comparison.png')
CM_FILE = os.path.join(OUTPUT_DIR, 'limfaad_model_comparison_confusion_matrices.png')


def load_metrics():
    data = {}
    for name, fname in METRICS_FILES.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[name] = dict(zip(df['metric'], df['value']))
        else:
            print(f"  Missing: {path}  (run the corresponding train script first)")
    return data


def load_confusion_matrices():
    """Reload validation result CSVs to rebuild confusion matrices."""
    cms = {}
    class_names = None

    val_files = {
        'XGBoost':        'limfaad_model_validation_results.csv',
        'Neural Network': 'limfaad_nn_validation_results.csv',
        'KNN':            'limfaad_knn_validation_results.csv',
    }

    # Label encoder for class names
    enc_candidates = [
        os.path.join(MODELS_DIR, 'limfaad_label_encoder.pkl'),
        os.path.join(MODELS_DIR, 'limfaad_knn_label_encoder.pkl'),
        os.path.join(MODELS_DIR, 'limfaad_nn_label_encoder.pkl'),
    ]
    for p in enc_candidates:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                enc = pickle.load(f)
            class_names = list(enc.classes_)
            break

    for name, fname in val_files.items():
        if fname is None:
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if 'true_label' in df.columns and 'predicted_label' in df.columns:
            labels = class_names if class_names else sorted(df['true_label'].unique())
            cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels)
            cms[name] = cm

    return cms, class_names


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("LIMFAAD: XGBoost vs Neural Network vs KNN comparison")
    print("=" * 70)

    # ── 1. Load metrics ────────────────────────────────────────────────────
    data = load_metrics()
    if not data:
        print("No metrics found. Run all three training scripts first.")
        return

    # ── 2. Print table ─────────────────────────────────────────────────────
    metrics_to_show = [
        ('test_accuracy',    'Test Accuracy'),
        ('macro_precision',  'Macro Precision'),
        ('macro_recall',     'Macro Recall'),
        ('macro_f1',         'Macro F1'),
        ('weighted_f1',      'Weighted F1'),
    ]
    models = list(data.keys())
    col_w = 15

    header = f"{'Metric':<25}" + "".join(f"{m:<{col_w}}" for m in models)
    separator = "-" * (25 + col_w * len(models))
    lines = ["=" * 70, "LIMFAAD: XGBoost vs Neural Network vs KNN (same train/test split)", "=" * 70,
             "", header, separator]

    for key, label in metrics_to_show:
        row = f"{label:<25}"
        for m in models:
            val = data[m].get(key, float('nan'))
            row += f"{val:<{col_w}.4f}"
        lines.append(row)

    # Best k for KNN
    if 'KNN' in data and 'best_k' in data['KNN']:
        lines.append(f"\n{'KNN best k':<25}{int(data['KNN']['best_k'])}")

    lines.append("\n" + "=" * 70)
    report = "\n".join(lines)
    print(report)

    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    print(f"\nSaved report: {REPORT_FILE}")

    # ── 3. Combined bar chart ──────────────────────────────────────────────
    plot_keys = [k for k, _ in metrics_to_show]
    plot_labels = [lbl for _, lbl in metrics_to_show]

    x = np.arange(len(plot_labels))
    n = len(models)
    w = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, m in enumerate(models):
        vals = [data[m].get(k, 0) for k in plot_keys]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=m,
                      color=COLORS.get(m, f'C{i}'), edgecolor='black', alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LIMFAAD Account Classification – XGBoost vs Neural Network vs KNN', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {CHART_FILE}")

    # ── 4. Side-by-side confusion matrices ────────────────────────────────
    cms, class_names = load_confusion_matrices()
    if cms and class_names:
        ncols = len(cms)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        for ax, (name, cm) in zip(axes, cms.items()):
            acc = np.trace(cm) / cm.sum()
            sns.heatmap(
                cm, annot=True, fmt='d',
                cmap='Blues' if name == 'XGBoost' else 'Oranges' if name == 'Neural Network' else 'Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax,
            )
            ax.set_title(f'{name}  (acc={acc:.3f})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        plt.suptitle('LIMFAAD – Confusion Matrices', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(CM_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrices: {CM_FILE}")

    print("\nDone.")


if __name__ == '__main__':
    main()
