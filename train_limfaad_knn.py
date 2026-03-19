"""
Task 3 (KNN variant): Train K-Nearest Neighbours on LIMFAAD for account classification.
Uses the exact same LIMFAAD data, preprocessing, and train/test split as XGBoost and BERT,
so all three models are directly comparable.

Includes:
  - Feature scaling (StandardScaler — required for KNN)
  - Grid search over k values to find the best k
  - Full evaluation: accuracy, per-class P/R/F1, confusion matrix
  - Saves model, scaler, metrics, report, and visualisations
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'input_file': 'LIMFADD.csv',
    'output_dir': 'task3_limfaad/outputs',
    'models_dir': 'task3_limfaad/models',
    'test_size': 0.2,
    'random_state': 42,
    # k values to search; best is selected by 5-fold CV on the training set
    'k_candidates': [3, 5, 7, 9, 11, 15, 21],
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)

FEATURE_COLUMNS = [
    'Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers',
    'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads',
]


# ---------------------------------------------------------------------------
# Preprocessing (same logic as train_limfaad_model.py and train_limfaad_bert.py)
# ---------------------------------------------------------------------------
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df_p = df.copy()

    df_p['Following/Followers'] = pd.to_numeric(df_p['Following/Followers'], errors='coerce')
    df_p['Posts/Followers'] = pd.to_numeric(df_p['Posts/Followers'], errors='coerce')

    df_p.loc[
        (df_p['Following/Followers'].isna()) & (df_p['Followers'] == 0),
        'Following/Followers'
    ] = df_p['Following'].max()
    df_p.loc[
        (df_p['Posts/Followers'].isna()) & (df_p['Followers'] == 0),
        'Posts/Followers'
    ] = 0
    df_p['Following/Followers'].fillna(df_p['Following/Followers'].median(), inplace=True)
    df_p['Posts/Followers'].fillna(df_p['Posts/Followers'].median(), inplace=True)

    for col in ['Bio', 'Profile Picture', 'External Link', 'Threads']:
        df_p[col] = df_p[col].apply(lambda x: 1 if str(x).lower() in ['yes', 'y'] else 0)

    return df_p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("TASK 3 (KNN): TRAIN K-NEAREST NEIGHBOURS ON LIMFAAD")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Model: KNeighborsClassifier (scikit-learn)")
    print("Dataset: LIMFADD.csv (same preprocessing and split as XGBoost & BERT)")
    print("=" * 80)

    # ── Step 1: Load & preprocess ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING LIMFAAD")
    print("=" * 80)
    df = load_and_preprocess(CONFIG['input_file'])
    print(f"Loaded {len(df)} samples")

    X = df[FEATURE_COLUMNS].copy()
    # Fill any remaining NaN with column median (KNN cannot handle NaN)
    X = X.fillna(X.median())
    # Safety fallback: fill any column that is entirely NaN with 0
    X = X.fillna(0)
    y = df['Labels'].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    print(f"Classes: {list(class_names)}")

    # ── Step 2: Same split as XGBoost and BERT ─────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING DATASET (same as XGBoost & BERT)")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y_encoded,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Step 3: Feature scaling (mandatory for KNN) ────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE SCALING (StandardScaler)")
    print("=" * 80)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Fitted scaler on training set, transformed train and test.")

    # ── Step 4: Find best k via 5-fold CV ──────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: FINDING BEST k (5-fold cross-validation on training set)")
    print("=" * 80)
    print(f"Searching k ∈ {CONFIG['k_candidates']} ...")

    cv_scores = {}
    for k in CONFIG['k_candidates']:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_scores[k] = scores.mean()
        print(f"  k={k:>3}  CV accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

    best_k = max(cv_scores, key=cv_scores.get)
    print(f"\nBest k = {best_k}  (CV accuracy = {cv_scores[best_k]:.4f})")

    # ── Step 5: Train final model with best k ──────────────────────────────
    print("\n" + "=" * 80)
    print(f"STEP 5: TRAINING KNN WITH k={best_k}")
    print("=" * 80)
    model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("Training done.")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATION ON TEST SET")
    print("=" * 80)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, labels=range(len(class_names))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='macro'
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted'
    )

    print(f"\nTraining Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nPer-class (test set):")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 58)
    for i, c in enumerate(class_names):
        print(f"{c:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    print(f"\nMacro  P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"Weighted P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f1:.4f}")

    # ── Step 7: Save model, scaler, label encoder ─────────────────────────
    print("\n" + "=" * 80)
    print("STEP 7: SAVING MODEL AND ARTIFACTS")
    print("=" * 80)

    model_path = os.path.join(CONFIG['models_dir'], 'limfaad_knn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    scaler_path = os.path.join(CONFIG['models_dir'], 'limfaad_knn_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler: {scaler_path}")

    # Save numpy arrays for fast inference without sklearn import
    np.save(os.path.join(CONFIG['models_dir'], 'limfaad_knn_X_train.npy'), X_train_scaled)
    np.save(os.path.join(CONFIG['models_dir'], 'limfaad_knn_y_train.npy'), y_train)
    np.save(os.path.join(CONFIG['models_dir'], 'limfaad_knn_scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(CONFIG['models_dir'], 'limfaad_knn_scaler_scale.npy'), scaler.scale_)
    print(f"Saved numpy inference arrays to {CONFIG['models_dir']}")

    encoder_path = os.path.join(CONFIG['models_dir'], 'limfaad_knn_label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder: {encoder_path}")

    feature_info = {'feature_columns': FEATURE_COLUMNS, 'class_names': class_names.tolist(), 'best_k': best_k}
    info_path = os.path.join(CONFIG['models_dir'], 'limfaad_knn_feature_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"Saved feature info: {info_path}")

    # ── Step 8: Save metrics and report ───────────────────────────────────
    metrics_df = pd.DataFrame({
        'metric': ['train_accuracy', 'test_accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                   'weighted_precision', 'weighted_recall', 'weighted_f1', 'best_k'],
        'value': [train_acc, test_acc, macro_p, macro_r, macro_f1,
                  weighted_p, weighted_r, weighted_f1, float(best_k)],
    })
    metrics_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # CV scores CSV
    cv_df = pd.DataFrame({'k': list(cv_scores.keys()), 'cv_accuracy': list(cv_scores.values())})
    cv_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_cv_scores.csv')
    cv_df.to_csv(cv_path, index=False)
    print(f"Saved CV scores: {cv_path}")

    # Validation results
    val_results = pd.DataFrame({
        'true_label': [class_names[y] for y in y_test],
        'predicted_label': [class_names[y] for y in y_test_pred],
        'confidence': [float(np.max(p)) for p in y_test_proba],
    })
    for i, c in enumerate(class_names):
        val_results[f'prob_{c.lower()}'] = y_test_proba[:, i]
    val_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_validation_results.csv')
    val_results.to_csv(val_path, index=False)
    print(f"Saved validation results: {val_path}")

    # Text report
    report_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_model_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LIMFAAD KNN MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: K-Nearest Neighbours (k={best_k}, metric=euclidean)\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {CONFIG['input_file']}\n")
        f.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n\n")
        f.write("Cross-validation k search:\n")
        for k, sc in cv_scores.items():
            marker = " ← best" if k == best_k else ""
            f.write(f"  k={k:>3}: {sc:.4f}{marker}\n")
        f.write(f"\nTraining Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy    : {test_acc:.4f}\n")
        f.write(f"Macro F1         : {macro_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_test_pred, target_names=class_names))
        cm = confusion_matrix(y_test, y_test_pred)
        f.write("\nConfusion Matrix (rows=true, cols=predicted):\n")
        f.write(str(cm) + "\n")
    print(f"Saved report: {report_path}")

    # ── Step 9: Visualisations ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 8: GENERATING VISUALISATIONS")
    print("=" * 80)

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'},
    )
    plt.title(f'LIMFAAD KNN (k={best_k}) – Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # k vs CV accuracy
    plt.figure(figsize=(8, 5))
    ks = list(cv_scores.keys())
    accs = list(cv_scores.values())
    plt.plot(ks, accs, marker='o', color='steelblue', linewidth=2, markersize=7)
    plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('k (number of neighbours)', fontsize=12)
    plt.ylabel('5-Fold CV Accuracy', fontsize=12)
    plt.title('KNN: CV Accuracy vs k', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    k_plot_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_k_selection.png')
    plt.savefig(k_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {k_plot_path}")

    # Per-class metrics bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(['Accuracy', 'Precision', 'Recall', 'F1'],
                [test_acc, macro_p, macro_r, macro_f1],
                color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], edgecolor='black')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title(f'KNN Overall Metrics (k={best_k})', fontsize=12, fontweight='bold')
    for i, v in enumerate([test_acc, macro_p, macro_r, macro_f1]):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)

    x = np.arange(len(class_names))
    w = 0.25
    axes[1].bar(x - w, precision, w, label='Precision', color='#3498db')
    axes[1].bar(x, recall, w, label='Recall', color='#e74c3c')
    axes[1].bar(x + w, f1, w, label='F1', color='#f39c12')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('KNN Per-Class Metrics', fontsize=12, fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    metrics_plot_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_metrics_visualization.png')
    plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {metrics_plot_path}")

    # Confidence distribution
    plt.figure(figsize=(8, 5))
    confidences = [float(np.max(p)) for p in y_test_proba]
    plt.hist(confidences, bins=20, color='mediumseagreen', edgecolor='black', alpha=0.8)
    plt.axvline(np.mean(confidences), color='red', linestyle='--',
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'KNN (k={best_k}) – Confidence Distribution (Test Set)')
    plt.legend()
    plt.tight_layout()
    conf_path = os.path.join(CONFIG['output_dir'], 'limfaad_knn_confidence_distribution.png')
    plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf_path}")

    print("\n" + "=" * 80)
    print("TASK 3 (KNN) COMPLETED")
    print("=" * 80)
    print(f"Best k         : {best_k}")
    print(f"Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Macro F1       : {macro_f1:.4f}")
    print("\nRun compare_limfaad_models.py to see XGBoost vs BERT vs KNN side-by-side.")
    print("=" * 80)


if __name__ == '__main__':
    main()
