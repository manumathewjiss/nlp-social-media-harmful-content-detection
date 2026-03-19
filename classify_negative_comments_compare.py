"""
Task 4 (Comparison): Classify negative-comment accounts with both XGBoost and BERT.
Uses LIMFAAD-trained XGBoost and LIMFAAD-trained BERT on the same Instagram
negative-comment profiles and outputs a side-by-side comparison.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

from limfaad_bert_utils import FEATURE_COLUMNS, row_to_text

# Optional: matplotlib for plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

CONFIG = {
    'input_file': 'task2_roberta/instagram_roberta_negative_comments.csv',
    'output_dir': 'task4_classification',
    'models_dir': 'task3_limfaad/models',
    'bert_model_dir': 'task3_limfaad/models/limfaad_bert',
    'max_length': 128,
    'bert_batch_size': 8,
}

# Apple Silicon MPS or CUDA or CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')
os.makedirs(CONFIG['output_dir'], exist_ok=True)


def parse_count(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(',', '').lower()
    if s in ('', 'nan', 'no', 'yes'):
        return np.nan
    try:
        if s.endswith('k'):
            return int(float(s[:-1]) * 1000)
        if s.endswith('m'):
            return int(float(s[:-1]) * 1_000_000)
        return int(float(s))
    except Exception:
        return np.nan


def parse_mutual_friends(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s == 'no':
        return 0
    if s == 'yes':
        return 1
    try:
        return int(float(s))
    except Exception:
        return 0


class TextInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


def main():
    print("=" * 80)
    print("TASK 4 (COMPARE): XGBoost vs BERT on negative-comment accounts")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ---- Load data ----
    print("\n" + "=" * 80)
    print("STEP 1: LOADING NEGATIVE COMMENTS USER PROFILES")
    print("=" * 80)
    df = pd.read_csv(CONFIG['input_file'])
    print(f"Loaded {len(df)} accounts")

    has_labels = 'Labels' in df.columns
    if has_labels:
        valid = df['Labels'].isin(['Bot', 'Scam', 'Real', 'Spam'])
        has_labels = valid.sum() == len(df)
    if has_labels:
        print("Ground-truth 'Labels' found. Will compute metrics for both models.")
    else:
        print("No ground-truth Labels. Comparison will use prediction distributions and agreement.")

    # ---- Preprocess (same as XGBoost script) ----
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING (match LIMFAAD format)")
    print("=" * 80)
    df['Followers'] = df['Followers'].apply(parse_count)
    df['Following'] = df['Following'].apply(parse_count)
    if 'Posts' not in df.columns:
        df['Posts'] = 0
    else:
        df['Posts'] = pd.to_numeric(df['Posts'], errors='coerce').fillna(0).astype(int)
    df['Followers'] = df['Followers'].fillna(0).astype(int)
    df['Following'] = df['Following'].fillna(0).astype(int)

    ff = np.where(
        df['Followers'].astype(float) > 0,
        df['Following'].astype(float) / np.maximum(df['Followers'].astype(float), 1e-9),
        np.nan,
    )
    df['Following/Followers'] = ff
    pf = np.where(
        df['Followers'].astype(float) > 0,
        df['Posts'].astype(float) / np.maximum(df['Followers'].astype(float), 1e-9),
        0.0,
    )
    df['Posts/Followers'] = pf
    df['Following/Followers'] = pd.to_numeric(df['Following/Followers'], errors='coerce')
    df['Posts/Followers'] = pd.to_numeric(df['Posts/Followers'], errors='coerce')
    ff_med = df['Following/Followers'].median()
    pf_med = df['Posts/Followers'].median()
    if np.isnan(ff_med):
        ff_med = 1.0
    if np.isnan(pf_med):
        pf_med = 0.0
    df['Following/Followers'] = df['Following/Followers'].fillna(ff_med)
    df['Posts/Followers'] = df['Posts/Followers'].fillna(pf_med)

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
    if 'Mutual_Friends' in df.columns:
        df['Mutual Friends'] = df['Mutual_Friends'].apply(parse_mutual_friends)
    else:
        df['Mutual Friends'] = 0
    print("Preprocessing done.")

    # ---- XGBoost ----
    print("\n" + "=" * 80)
    print("STEP 3: XGBoost INFERENCE")
    print("=" * 80)
    model_path = os.path.join(CONFIG['models_dir'], 'limfaad_xgboost_model.json')
    encoder_path = os.path.join(CONFIG['models_dir'], 'limfaad_label_encoder.pkl')
    info_path = os.path.join(CONFIG['models_dir'], 'limfaad_feature_info.pkl')
    if not os.path.exists(model_path):
        print(f"Missing {model_path}. Run train_limfaad_model.py first.")
        return
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder_xgb = pickle.load(f)
    with open(info_path, 'rb') as f:
        feature_info = pickle.load(f)
    feature_columns = feature_info['feature_columns']
    class_names = np.array(feature_info['class_names'])

    X = df[feature_columns].copy().fillna(0)
    pred_xgb = model_xgb.predict(X)
    proba_xgb = model_xgb.predict_proba(X)
    labels_xgb = label_encoder_xgb.inverse_transform(pred_xgb)
    df['Predicted_Label_XGBoost'] = labels_xgb
    df['Confidence_XGBoost'] = [float(np.max(p)) for p in proba_xgb]
    print(f"XGBoost predictions: {pd.Series(labels_xgb).value_counts().to_dict()}")

    # ---- BERT ----
    print("\n" + "=" * 80)
    print("STEP 4: BERT INFERENCE")
    print("=" * 80)
    bert_dir = CONFIG['bert_model_dir']
    if not os.path.exists(os.path.join(bert_dir, 'config.json')):
        print(f"BERT model not found at {bert_dir}. Run train_limfaad_bert.py first.")
        df['Predicted_Label_BERT'] = None
        df['Confidence_BERT'] = np.nan
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
        model_bert = BertForSequenceClassification.from_pretrained(bert_dir)
        with open(os.path.join(CONFIG['models_dir'], 'limfaad_bert_label_encoder.pkl'), 'rb') as f:
            label_encoder_bert = pickle.load(f)
        model_bert.eval()
        device = get_device()
        print(f"Using device: {device}")
        model_bert.to(device)

        texts = [row_to_text(df.loc[i]) for i in range(len(df))]
        dataset = TextInferenceDataset(texts, tokenizer, CONFIG['max_length'])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=CONFIG['bert_batch_size'],
            shuffle=False,
            num_workers=0,
        )
        all_logits = []
        print(f"Running BERT inference on {len(texts)} accounts ({len(loader)} batches)...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="BERT inference"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                out = model_bert(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(out.logits.cpu().numpy())
        logits = np.vstack(all_logits)
        pred_bert = np.argmax(logits, axis=1)
        proba_bert = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
        labels_bert = label_encoder_bert.inverse_transform(pred_bert)
        df['Predicted_Label_BERT'] = labels_bert
        df['Confidence_BERT'] = [float(np.max(p)) for p in proba_bert]
        print(f"BERT predictions: {pd.Series(labels_bert).value_counts().to_dict()}")

    # ---- KNN (pure NumPy — no sklearn import at runtime) ----
    print("\n" + "=" * 80)
    print("STEP 5: KNN INFERENCE (NumPy)")
    print("=" * 80)
    knn_X_train_path  = os.path.join(CONFIG['models_dir'], 'limfaad_knn_X_train.npy')
    knn_y_train_path  = os.path.join(CONFIG['models_dir'], 'limfaad_knn_y_train.npy')
    knn_mean_path     = os.path.join(CONFIG['models_dir'], 'limfaad_knn_scaler_mean.npy')
    knn_scale_path    = os.path.join(CONFIG['models_dir'], 'limfaad_knn_scaler_scale.npy')
    knn_info_path     = os.path.join(CONFIG['models_dir'], 'limfaad_knn_feature_info.pkl')

    if not os.path.exists(knn_X_train_path):
        print("KNN numpy arrays not found. Run train_limfaad_knn.py first.")
        df['Predicted_Label_KNN'] = None
        df['Confidence_KNN'] = np.nan
    else:
        X_train_knn = np.load(knn_X_train_path)
        y_train_knn = np.load(knn_y_train_path)
        scaler_mean = np.load(knn_mean_path)
        scaler_scale = np.load(knn_scale_path)
        with open(knn_info_path, 'rb') as f:
            knn_info = pickle.load(f)
        knn_features = knn_info['feature_columns']
        class_names_knn = np.array(knn_info['class_names'])
        best_k = int(knn_info['best_k'])

        # Scale test features using saved scaler parameters
        X_knn = df[knn_features].copy().fillna(0).values.astype(float)
        X_knn_scaled = (X_knn - scaler_mean) / scaler_scale

        # Pure NumPy KNN inference
        n_classes = len(class_names_knn)
        pred_knn = []
        proba_knn = []
        print(f"Running KNN (k={best_k}) inference on {len(X_knn_scaled)} accounts...")
        for x in X_knn_scaled:
            dists = np.sqrt(np.sum((X_train_knn - x) ** 2, axis=1))
            k_idx = np.argsort(dists)[:best_k]
            k_labels = y_train_knn[k_idx]
            counts = np.bincount(k_labels, minlength=n_classes)
            proba = counts / best_k
            pred_knn.append(np.argmax(counts))
            proba_knn.append(proba)
        pred_knn = np.array(pred_knn)
        proba_knn = np.array(proba_knn)
        labels_knn = class_names_knn[pred_knn]
        df['Predicted_Label_KNN'] = labels_knn
        df['Confidence_KNN'] = [float(np.max(p)) for p in proba_knn]
        print(f"KNN predictions: {pd.Series(labels_knn).value_counts().to_dict()}")

    # ---- Agreement and comparison ----
    bert_ready = 'Predicted_Label_BERT' in df.columns and df['Predicted_Label_BERT'].notna().all()
    knn_ready  = 'Predicted_Label_KNN'  in df.columns and df['Predicted_Label_KNN'].notna().all()
    if bert_ready:
        agreement_xgb_bert = (df['Predicted_Label_XGBoost'] == df['Predicted_Label_BERT']).mean()
        print(f"\nXGBoost–BERT agreement: {agreement_xgb_bert:.2%} ({int(agreement_xgb_bert*len(df))}/{len(df)})")
        df['Agreement_XGB_BERT'] = df['Predicted_Label_XGBoost'] == df['Predicted_Label_BERT']
    if knn_ready:
        agreement_xgb_knn = (df['Predicted_Label_XGBoost'] == df['Predicted_Label_KNN']).mean()
        print(f"XGBoost–KNN agreement : {agreement_xgb_knn:.2%} ({int(agreement_xgb_knn*len(df))}/{len(df)})")
        df['Agreement_XGB_KNN'] = df['Predicted_Label_XGBoost'] == df['Predicted_Label_KNN']
    if bert_ready and knn_ready:
        agreement_bert_knn = (df['Predicted_Label_BERT'] == df['Predicted_Label_KNN']).mean()
        print(f"BERT–KNN agreement    : {agreement_bert_knn:.2%} ({int(agreement_bert_knn*len(df))}/{len(df)})")

    # ---- Metrics if labels exist ----
    if has_labels:
        y_true = df['Labels'].values
        print("\n" + "=" * 80)
        print("METRICS (ground-truth Labels)")
        print("=" * 80)
        for name, pred_col in [
            ('XGBoost', 'Predicted_Label_XGBoost'),
            ('BERT',    'Predicted_Label_BERT'),
            ('KNN',     'Predicted_Label_KNN'),
        ]:
            if pred_col not in df.columns or df[pred_col].isna().any():
                continue
            y_pred = df[pred_col].values
            acc = accuracy_score(y_true, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=class_names, average='macro'
            )
            print(f"\n{name}: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}")
            print(classification_report(y_true, y_pred, labels=class_names, target_names=class_names))

    # ---- Save results ----
    print("\n" + "=" * 80)
    print("STEP 5: SAVING RESULTS")
    print("=" * 80)
    out_cols = [
        c for c in
        ['Post_ID', 'Post_Text', 'User_Name', 'Followers', 'Following', 'Posts',
         'Predicted_Label_XGBoost', 'Confidence_XGBoost',
         'Predicted_Label_BERT',    'Confidence_BERT',
         'Predicted_Label_KNN',     'Confidence_KNN',
         'Agreement_XGB_BERT', 'Agreement_XGB_KNN', 'Labels']
        if c in df.columns
    ]
    out = df[out_cols].copy()
    results_file = os.path.join(CONFIG['output_dir'], 'instagram_negative_xgboost_vs_bert_vs_knn_results.csv')
    out.to_csv(results_file, index=False)
    print(f"Saved: {results_file}")

    report_file = os.path.join(CONFIG['output_dir'], 'instagram_negative_xgboost_vs_bert_vs_knn_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TASK 4: XGBoost vs BERT vs KNN – Account classification comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input: {CONFIG['input_file']}\n")
        f.write(f"Total accounts: {len(df)}\n\n")
        for name, col in [
            ('XGBoost', 'Predicted_Label_XGBoost'),
            ('BERT',    'Predicted_Label_BERT'),
            ('KNN',     'Predicted_Label_KNN'),
        ]:
            if col not in df.columns or df[col].isna().any():
                continue
            f.write(f"{name} distribution:\n")
            for k, v in df[col].value_counts().reindex(class_names, fill_value=0).items():
                f.write(f"  {k}: {v} ({100*v/len(df):.1f}%)\n")
            f.write("\n")
        if bert_ready:
            f.write(f"XGBoost–BERT agreement: {agreement_xgb_bert:.2%}\n")
        if knn_ready:
            f.write(f"XGBoost–KNN agreement : {agreement_xgb_knn:.2%}\n")
        if bert_ready and knn_ready:
            f.write(f"BERT–KNN agreement    : {agreement_bert_knn:.2%}\n")
    print(f"Saved: {report_file}")

    # ---- Comparison plot (all available models) ----
    if HAS_MPL:
        models_to_plot = [
            ('XGBoost', 'Predicted_Label_XGBoost', '#3498db'),
            ('BERT',    'Predicted_Label_BERT',    '#e74c3c'),
            ('KNN',     'Predicted_Label_KNN',     '#2ecc71'),
        ]
        available = [(n, c, col) for n, c, col in models_to_plot
                     if c in df.columns and df[c].notna().all()]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
            if len(available) == 1:
                axes = [axes]
            bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            for ax, (name, col, _) in zip(axes, available):
                dist = df[col].value_counts().reindex(class_names, fill_value=0)
                bars = ax.bar(dist.index, dist.values, color=bar_colors, edgecolor='black')
                for bar, v in zip(bars, dist.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            str(int(v)), ha='center', fontweight='bold')
                ax.set_title(f'{name} predictions', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count')
                ax.set_ylim(0, max(dist.values) + 5)
            plt.suptitle('XGBoost vs BERT vs KNN – Instagram negative-comment account classification',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plot_path = os.path.join(CONFIG['output_dir'], 'instagram_negative_xgboost_vs_bert_vs_knn_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")

    print("\n" + "=" * 80)
    print("TASK 4 (COMPARE) COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
