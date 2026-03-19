"""
Task 3 (BERT variant): Train BERT on LIMFAAD for account classification.
Uses the same LIMFAAD data and train/test split as XGBoost, but converts
each row to text and fine-tunes BERT for 4-class classification (Bot, Scam, Real, Spam).
Enables direct comparison: XGBoost (tabular) vs BERT (tabular-as-text).
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None
import warnings
warnings.filterwarnings('ignore')

from limfaad_bert_utils import FEATURE_COLUMNS, row_to_text

# ---------------------------------------------------------------------------
# Config (aligned with train_limfaad_model.py for same split)
# ---------------------------------------------------------------------------
CONFIG = {
    'input_file': 'LIMFADD.csv',
    'output_dir': 'task3_limfaad/outputs',
    'models_dir': 'task3_limfaad/models',
    'bert_model_dir': 'task3_limfaad/models/limfaad_bert',
    'test_size': 0.2,
    'random_state': 42,
    'bert_model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)
os.makedirs(CONFIG['bert_model_dir'], exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocess LIMFAAD (same logic as train_limfaad_model.py)
# ---------------------------------------------------------------------------
def load_and_preprocess_limfaad(csv_path: str):
    df = pd.read_csv(csv_path)
    df_processed = df.copy()

    # Ratio columns: handle #DIV/0!
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

    # Encode categoricals
    df_processed['Bio'] = df_processed['Bio'].apply(
        lambda x: 1 if str(x).lower() in ['yes', 'y'] else 0
    )
    df_processed['Profile Picture'] = df_processed['Profile Picture'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )
    df_processed['External Link'] = df_processed['External Link'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )
    df_processed['Threads'] = df_processed['Threads'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )

    return df_processed


# ---------------------------------------------------------------------------
# PyTorch dataset for BERT
# ---------------------------------------------------------------------------
class LimfaadTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("TASK 3 (BERT): TRAIN BERT ON LIMFAAD FOR ACCOUNT CLASSIFICATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Model: BERT (tabular rows converted to text)")
    print("Dataset: LIMFADD.csv (same as XGBoost)")
    print("=" * 80)

    # 1) Load and preprocess
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING LIMFAAD")
    print("=" * 80)
    df = load_and_preprocess_limfaad(CONFIG['input_file'])
    print(f"Loaded {len(df)} samples")

    # 2) Build text and labels
    texts = [row_to_text(df.loc[i]) for i in range(len(df))]
    y = df['Labels'].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    print(f"Classes: {list(class_names)}")

    # 3) Same split as XGBoost (stratified, random_state=42)
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y_encoded,
    )
    train_texts = [texts[i] for i in idx_train]
    train_labels = [y_encoded[i] for i in idx_train]
    test_texts = [texts[i] for i in idx_test]
    test_labels = [y_encoded[i] for i in idx_test]
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # 4) Tokenizer and datasets
    print("\n" + "=" * 80)
    print("STEP 2: TOKENIZING AND CREATING DATASETS")
    print("=" * 80)
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model_name'])
    train_ds = LimfaadTextDataset(
        train_texts, train_labels, tokenizer, CONFIG['max_length']
    )
    test_ds = LimfaadTextDataset(
        test_texts, test_labels, tokenizer, CONFIG['max_length']
    )

    # 5) Model
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['bert_model_name'],
        num_labels=len(class_names),
    )

    # 6) Training args and Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(CONFIG['output_dir'], 'bert_checkpoints'),
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        warmup_ratio=CONFIG['warmup_ratio'],
        weight_decay=CONFIG['weight_decay'],
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=1,
        report_to='none',
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    callbacks = []
    if EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("\n" + "=" * 80)
    print("STEP 3: TRAINING BERT")
    print("=" * 80)
    trainer.train()

    # 7) Evaluate on test set
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATION ON TEST SET")
    print("=" * 80)
    test_pred = trainer.predict(test_ds)
    y_test_pred = np.argmax(test_pred.predictions, axis=-1)
    y_test = test_labels

    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, labels=range(len(class_names))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='macro'
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted'
    )

    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class (test set):")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 58)
    for i, c in enumerate(class_names):
        print(f"{c:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

    # 8) Save model, tokenizer, label encoder, feature info
    print("\n" + "=" * 80)
    print("STEP 5: SAVING MODEL AND ARTIFACTS")
    print("=" * 80)
    model.save_pretrained(CONFIG['bert_model_dir'])
    tokenizer.save_pretrained(CONFIG['bert_model_dir'])

    encoder_path = os.path.join(CONFIG['models_dir'], 'limfaad_bert_label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder: {encoder_path}")

    feature_info = {
        'feature_columns': FEATURE_COLUMNS,
        'class_names': class_names.tolist(),
    }
    info_path = os.path.join(CONFIG['models_dir'], 'limfaad_bert_feature_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"Saved feature info: {info_path}")

    # 9) Save report and metrics
    report_path = os.path.join(CONFIG['output_dir'], 'limfaad_bert_model_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LIMFAAD BERT MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: BERT ({CONFIG['bert_model_name']})\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {CONFIG['input_file']}\n")
        f.write(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_test_pred, target_names=class_names))
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        cm = confusion_matrix(y_test, y_test_pred)
        f.write(str(cm) + "\n")
    print(f"Saved report: {report_path}")

    metrics_df = pd.DataFrame({
        'metric': ['test_accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                  'weighted_precision', 'weighted_recall', 'weighted_f1'],
        'value': [test_accuracy, macro_p, macro_r, macro_f1,
                  weighted_p, weighted_r, weighted_f1],
    })
    metrics_path = os.path.join(CONFIG['output_dir'], 'limfaad_bert_training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # Save validation results CSV (same format as XGBoost and KNN)
    val_proba = torch.softmax(
        torch.tensor(test_pred.predictions, dtype=torch.float32), dim=1
    ).numpy()
    val_results = pd.DataFrame({
        'true_label':      [class_names[y] for y in y_test],
        'predicted_label': [class_names[y] for y in y_test_pred],
        'confidence':      [float(np.max(p)) for p in val_proba],
    })
    for i, c in enumerate(class_names):
        val_results[f'prob_{c.lower()}'] = val_proba[:, i]
    val_results_path = os.path.join(CONFIG['output_dir'], 'limfaad_bert_validation_results.csv')
    val_results.to_csv(val_results_path, index=False)
    print(f"Saved validation results: {val_results_path}")

    # Confusion matrix plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_test, y_test_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'},
        )
        plt.title('LIMFAAD BERT – Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(CONFIG['output_dir'], 'limfaad_bert_confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix: {cm_path}")
    except Exception as e:
        print(f"Skipping plot: {e}")

    print("\n" + "=" * 80)
    print("TASK 3 (BERT) COMPLETED")
    print("=" * 80)
    print(f"Test Accuracy: {test_accuracy:.4f}  Macro F1: {macro_f1:.4f}")
    print("Run classify_negative_comments_compare.py to compare XGBoost vs BERT on Instagram accounts.")
    print("=" * 80)


if __name__ == '__main__':
    main()
