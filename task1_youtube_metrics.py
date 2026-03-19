import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from preprocess import clean_text
import numpy as np


class Config:
    INPUT_CSV = "YoutubeCommentsDataSet.csv"
    OUTPUT_DIR = "outputs"
    METRICS_REPORT = "youtube_comments_metrics_report.txt"
    CONFUSION_MATRIX_IMG = "youtube_comments_confusion_matrix.png"
    METRICS_VISUALIZATION = "youtube_comments_metrics_visualization.png"
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    CACHE_DIR = "./model_cache"
    BATCH_SIZE = 16
    MAX_LENGTH = 128


def load_dataset(file_path):
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Successfully loaded {len(df):,} comments")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nFirst 3 comments preview:")
    for i in range(min(3, len(df))):
        comment = df.loc[i, 'Comment'][:80]
        sentiment = df.loc[i, 'Sentiment']
        print(f"  {i+1}. [{sentiment}] {comment}...")
    
    print(f"\nOriginal sentiment distribution:")
    sentiment_counts = df['Sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize():10s}: {count:5,} ({percentage:5.2f}%)")
    
    return df


def preprocess_data(df):
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING COMMENTS")
    print("="*70)
    
    print(f"Cleaning {len(df):,} comments...")
    print("   (Removing URLs, mentions, hashtags, emojis, special characters)")
    
    tqdm.pandas(desc="Cleaning")
    df['clean_text'] = df['Comment'].progress_apply(clean_text)
    
    initial_count = len(df)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"Removed {removed} empty comments after cleaning")
    
    print(f"Preprocessing complete! {len(df):,} valid comments ready")
    
    return df


def load_model():
    print("\n" + "="*70)
    print("STEP 3: LOADING ROBERTA MODEL")
    print("="*70)
    
    print(f"Model: {Config.MODEL_NAME}")
    print("   (This will download ~500 MB on first run)")
    print("   (Subsequent runs will use cached model)")
    print(f"   Cache directory: {Config.CACHE_DIR}")
    
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    print("Tokenizer loaded")
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    print("Model loaded")
    
    labels = ['negative', 'neutral', 'positive']
    
    print(f"\nModel Info:")
    print(f"   Architecture: RoBERTa (Robustly Optimized BERT)")
    print(f"   Training: 124M tweets (2018-2021)")
    print(f"   Input: Text (max {Config.MAX_LENGTH} tokens)")
    print(f"   Output: One of {labels}")
    print(f"   Batch size: {Config.BATCH_SIZE} comments at once")
    
    return tokenizer, model, labels


def analyze_sentiment(df, tokenizer, model, labels):
    print("\n" + "="*70)
    print("STEP 4: ANALYZING SENTIMENT WITH ROBERTA")
    print("="*70)
    
    print(f"Processing {len(df):,} comments in batches of {Config.BATCH_SIZE}")
    print("   This will take several minutes...")
    
    texts = df['clean_text'].tolist()
    results = []
    
    model.eval()
    
    total_batches = (len(texts) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    print(f"   Total batches: {total_batches:,}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), Config.BATCH_SIZE), 
                     desc="Analyzing", 
                     unit="batch"):
            
            batch = texts[i:i+Config.BATCH_SIZE]
            
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
            
            outputs = model(**tokens)
            probs = F.softmax(outputs.logits, dim=1)
            predictions = [labels[p.argmax().item()] for p in probs]
            
            results.extend(predictions)
    
    df['Predicted_Sentiment'] = results
    
    print(f"\nSentiment analysis complete!")
    print(f"   Processed {len(df):,} comments")
    
    print(f"\nPredicted sentiment distribution:")
    pred_counts = df['Predicted_Sentiment'].value_counts()
    for sentiment, count in pred_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize():10s}: {count:5,} ({percentage:5.2f}%)")
    
    return df


def calculate_metrics(y_true, y_pred, labels):
    """
    Calculate per-class and macro-averaged F1, Precision, and Recall
    """
    # Get per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Calculate macro-averaged metrics
    macro_precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # Calculate weighted-averaged metrics (for reference)
    weighted_precision = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    
    metrics = {
        'per_class': {
            'precision': dict(zip(labels, precision_per_class)),
            'recall': dict(zip(labels, recall_per_class)),
            'f1': dict(zip(labels, f1_per_class)),
            'support': dict(zip(labels, support))
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    }
    
    return metrics


def evaluate_model(df):
    print("\n" + "="*70)
    print("STEP 5: CALCULATING METRICS")
    print("="*70)
    
    y_true = df['Sentiment']
    y_pred = df['Predicted_Sentiment']
    labels = ['negative', 'neutral', 'positive']
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    # Calculate detailed metrics
    metrics = calculate_metrics(y_true, y_pred, labels)
    
    # Print per-class metrics
    print(f"\n{'='*70}")
    print("PER-CLASS METRICS")
    print(f"{'='*70}")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 70)
    for label in labels:
        prec = metrics['per_class']['precision'][label]
        rec = metrics['per_class']['recall'][label]
        f1 = metrics['per_class']['f1'][label]
        sup = metrics['per_class']['support'][label]
        print(f"{label.capitalize():<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {int(sup):<12}")
    
    # Print macro-averaged metrics
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED METRICS")
    print(f"{'='*70}")
    print(f"Macro Precision: {metrics['macro']['precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro']['recall']:.4f}")
    print(f"Macro F1-Score:  {metrics['macro']['f1']:.4f}")
    
    # Print weighted-averaged metrics (for reference)
    print(f"\n{'='*70}")
    print("WEIGHTED-AVERAGED METRICS (Reference)")
    print(f"{'='*70}")
    print(f"Weighted Precision: {metrics['weighted']['precision']:.4f}")
    print(f"Weighted Recall:    {metrics['weighted']['recall']:.4f}")
    print(f"Weighted F1-Score:  {metrics['weighted']['f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX")
    print(f"{'='*70}")
    print("   Rows = Actual, Columns = Predicted")
    print("   Diagonal = Correct predictions")
    print("\n" + str(cm))
    
    # Add confusion matrix and accuracy to metrics
    metrics['accuracy'] = accuracy
    metrics['confusion_matrix'] = cm
    metrics['classification_report'] = classification_report(
        y_true, y_pred, 
        target_names=labels,
        digits=4
    )
    
    return metrics


def create_visualizations(metrics, labels):
    print("\n" + "="*70)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[l.capitalize() for l in labels],
                yticklabels=[l.capitalize() for l in labels])
    plt.title('Confusion Matrix\nRoBERTa on YouTube Comments Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.tight_layout()
    
    confusion_path = os.path.join(Config.OUTPUT_DIR, Config.CONFUSION_MATRIX_IMG)
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved: {confusion_path}")
    plt.close()
    
    # 2. Metrics Visualization (Bar chart for per-class and macro metrics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare data
    class_names = [l.capitalize() for l in labels] + ['Macro-Avg']
    precision_values = [metrics['per_class']['precision'][l] for l in labels] + [metrics['macro']['precision']]
    recall_values = [metrics['per_class']['recall'][l] for l in labels] + [metrics['macro']['recall']]
    f1_values = [metrics['per_class']['f1'][l] for l in labels] + [metrics['macro']['f1']]
    
    x_pos = np.arange(len(class_names))
    width = 0.6
    
    # Precision plot
    axes[0].bar(x_pos, precision_values, width, color='#3498db', alpha=0.8)
    axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[0].set_title('Precision by Class', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precision_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall plot
    axes[1].bar(x_pos, recall_values, width, color='#2ecc71', alpha=0.8)
    axes[1].set_ylabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_title('Recall by Class', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recall_values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score plot
    axes[2].bar(x_pos, f1_values, width, color='#e74c3c', alpha=0.8)
    axes[2].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[2].set_title('F1-Score by Class', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_values):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('RoBERTa Performance Metrics on YouTube Comments Dataset', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    metrics_path = os.path.join(Config.OUTPUT_DIR, Config.METRICS_VISUALIZATION)
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"Metrics visualization saved: {metrics_path}")
    plt.close()


def save_metrics_report(df, metrics, labels):
    print("\n" + "="*70)
    print("STEP 7: SAVING METRICS REPORT")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(Config.OUTPUT_DIR, Config.METRICS_REPORT)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("YOUTUBE COMMENTS DATASET - ROBERTA METRICS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {Config.INPUT_CSV}\n")
        f.write(f"Total Comments: {len(df):,}\n")
        f.write(f"Model: {Config.MODEL_NAME}\n\n")
        
        # Overall accuracy
        f.write(f"{'='*70}\n")
        f.write("OVERALL ACCURACY\n")
        f.write(f"{'='*70}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n\n")
        
        # Per-class metrics
        f.write(f"{'='*70}\n")
        f.write("PER-CLASS METRICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
        f.write("-" * 70 + "\n")
        for label in labels:
            prec = metrics['per_class']['precision'][label]
            rec = metrics['per_class']['recall'][label]
            f1 = metrics['per_class']['f1'][label]
            sup = metrics['per_class']['support'][label]
            f.write(f"{label.capitalize():<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {int(sup):<12}\n")
        
        # Macro-averaged metrics
        f.write(f"\n{'='*70}\n")
        f.write("MACRO-AVERAGED METRICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Macro Precision: {metrics['macro']['precision']:.4f}\n")
        f.write(f"Macro Recall:    {metrics['macro']['recall']:.4f}\n")
        f.write(f"Macro F1-Score:  {metrics['macro']['f1']:.4f}\n")
        
        # Weighted-averaged metrics
        f.write(f"\n{'='*70}\n")
        f.write("WEIGHTED-AVERAGED METRICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Weighted Precision: {metrics['weighted']['precision']:.4f}\n")
        f.write(f"Weighted Recall:    {metrics['weighted']['recall']:.4f}\n")
        f.write(f"Weighted F1-Score:  {metrics['weighted']['f1']:.4f}\n")
        
        # Confusion matrix
        f.write(f"\n{'='*70}\n")
        f.write("CONFUSION MATRIX\n")
        f.write(f"{'='*70}\n")
        f.write("   Rows = Actual, Columns = Predicted\n")
        f.write("   Diagonal = Correct predictions\n\n")
        cm = metrics['confusion_matrix']
        f.write("           ")
        for label in labels:
            f.write(f"{label.capitalize():<12}")
        f.write("\n")
        for i, label in enumerate(labels):
            f.write(f"{label.capitalize():<12}")
            for j in range(len(labels)):
                f.write(f"{cm[i][j]:<12}")
            f.write("\n")
        
        # Detailed classification report
        f.write(f"\n{'='*70}\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write(f"{'='*70}\n")
        f.write(metrics['classification_report'])
        
        # Dataset distribution
        f.write(f"\n{'='*70}\n")
        f.write("DATASET DISTRIBUTION\n")
        f.write(f"{'='*70}\n")
        sentiment_counts = df['Sentiment'].value_counts()
        for sentiment in labels:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment.capitalize():<12}: {count:>6,} ({percentage:>6.2f}%)\n")
    
    print(f"Metrics report saved: {report_path}")
    print(f"\nAll output files are in: {Config.OUTPUT_DIR}/")
    print(f"  - {Config.METRICS_REPORT}")
    print(f"  - {Config.CONFUSION_MATRIX_IMG}")
    print(f"  - {Config.METRICS_VISUALIZATION}")


def display_summary(df, metrics, start_time):
    print("\n" + "="*70)
    print("TASK 1 COMPLETE!")
    print("="*70)
    
    runtime = (datetime.now() - start_time).total_seconds()
    minutes = int(runtime // 60)
    seconds = int(runtime % 60)
    
    print(f"\nTotal Runtime: {minutes}m {seconds}s")
    print(f"Comments Analyzed: {len(df):,}")
    print(f"Model Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    
    print(f"\nMacro-Averaged Metrics:")
    print(f"   Precision: {metrics['macro']['precision']:.4f}")
    print(f"   Recall:    {metrics['macro']['recall']:.4f}")
    print(f"   F1-Score:  {metrics['macro']['f1']:.4f}")
    
    print(f"\nOutput Files:")
    print(f"   - {Config.OUTPUT_DIR}/{Config.METRICS_REPORT}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.CONFUSION_MATRIX_IMG}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.METRICS_VISUALIZATION}")
    
    print("\n" + "="*70)
    print("Task 1 completed successfully!")
    print("="*70 + "\n")


def main():
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("TASK 1: YOUTUBE COMMENTS DATASET METRICS")
    print("MODEL: ROBERTA-TWITTER")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: RoBERTa-Twitter (cardiffnlp)")
    print(f"Dataset: {Config.INPUT_CSV}")
    print("="*70)
    
    try:
        df = load_dataset(Config.INPUT_CSV)
        df = preprocess_data(df)
        tokenizer, model, labels = load_model()
        df = analyze_sentiment(df, tokenizer, model, labels)
        metrics = evaluate_model(df)
        create_visualizations(metrics, labels)
        save_metrics_report(df, metrics, labels)
        display_summary(df, metrics, start_time)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print(f"\nTip: Make sure {Config.INPUT_CSV} is in the same folder!")
        raise


if __name__ == "__main__":
    main()
