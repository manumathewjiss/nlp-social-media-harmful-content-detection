import os
import pandas as pd
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
import numpy as np


class Config:
    INPUT_CSV = "outputs/phase1_validation_results.csv"
    OUTPUT_DIR = "outputs"
    METRICS_REPORT = "task1_collection_metrics_report.txt"
    CONFUSION_MATRIX_IMG = "task1_collection_confusion_matrix.png"
    METRICS_VISUALIZATION = "task1_collection_metrics_visualization.png"


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
    
    # Check if required columns exist
    if 'Ground_Truth_Sentiment' not in df.columns:
        raise ValueError("Missing 'Ground_Truth_Sentiment' column")
    if 'Predicted_Sentiment' not in df.columns:
        raise ValueError("Missing 'Predicted_Sentiment' column")
    
    print(f"\nFirst 3 comments preview:")
    for i in range(min(3, len(df))):
        comment = df.loc[i, 'Comment_Text'][:80] if 'Comment_Text' in df.columns else "N/A"
        true_sent = df.loc[i, 'Ground_Truth_Sentiment']
        pred_sent = df.loc[i, 'Predicted_Sentiment']
        print(f"  {i+1}. [{true_sent}→{pred_sent}] {comment}...")
    
    print(f"\nGround truth sentiment distribution:")
    true_counts = df['Ground_Truth_Sentiment'].value_counts()
    for sentiment, count in true_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize():10s}: {count:5,} ({percentage:5.2f}%)")
    
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
    print("STEP 2: CALCULATING METRICS")
    print("="*70)
    
    y_true = df['Ground_Truth_Sentiment']
    y_pred = df['Predicted_Sentiment']
    
    # Get unique labels from both true and predicted
    all_labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    labels = ['negative', 'neutral', 'positive']  # Standard order
    
    # Filter to only include labels that exist in the data
    labels = [l for l in labels if l in all_labels]
    
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
    metrics['labels'] = labels
    metrics['classification_report'] = classification_report(
        y_true, y_pred, 
        target_names=labels,
        digits=4
    )
    
    return metrics


def create_visualizations(metrics, labels):
    print("\n" + "="*70)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[l.capitalize() for l in labels],
                yticklabels=[l.capitalize() for l in labels])
    plt.title('Confusion Matrix\nRoBERTa on Task 1 Collection Template Dataset', fontsize=14, fontweight='bold')
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
    
    plt.suptitle('RoBERTa Performance Metrics on Task 1 Collection Template Dataset', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    metrics_path = os.path.join(Config.OUTPUT_DIR, Config.METRICS_VISUALIZATION)
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"Metrics visualization saved: {metrics_path}")
    plt.close()


def save_metrics_report(df, metrics, labels):
    print("\n" + "="*70)
    print("STEP 4: SAVING METRICS REPORT")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(Config.OUTPUT_DIR, Config.METRICS_REPORT)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TASK 1 COLLECTION TEMPLATE DATASET - ROBERTA METRICS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {Config.INPUT_CSV}\n")
        f.write(f"Total Comments: {len(df):,}\n")
        f.write(f"Model: RoBERTa-Twitter (cardiffnlp/twitter-roberta-base-sentiment)\n\n")
        
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
        f.write("Ground Truth Distribution:\n")
        true_counts = df['Ground_Truth_Sentiment'].value_counts()
        for sentiment in labels:
            count = true_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment.capitalize():<12}: {count:>6,} ({percentage:>6.2f}%)\n")
        
        f.write("\nPredicted Distribution:\n")
        pred_counts = df['Predicted_Sentiment'].value_counts()
        for sentiment in labels:
            count = pred_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment.capitalize():<12}: {count:>6,} ({percentage:>6.2f}%)\n")
        
        # Note about dataset
        f.write(f"\n{'='*70}\n")
        f.write("NOTE\n")
        f.write(f"{'='*70}\n")
        f.write("This dataset contains manually collected comments that were all labeled as 'negative'.\n")
        f.write("The metrics show how well RoBERTa identifies these negative comments.\n")
        f.write("Since all ground truth labels are 'negative', the negative class metrics are most meaningful.\n")
    
    print(f"Metrics report saved: {report_path}")
    print(f"\nAll output files are in: {Config.OUTPUT_DIR}/")
    print(f"  - {Config.METRICS_REPORT}")
    print(f"  - {Config.CONFUSION_MATRIX_IMG}")
    print(f"  - {Config.METRICS_VISUALIZATION}")


def display_summary(df, metrics, start_time):
    print("\n" + "="*70)
    print("TASK 1 COLLECTION ANALYSIS COMPLETE!")
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
    
    # Show negative class performance (most relevant since all are negative)
    if 'negative' in metrics['labels']:
        print(f"\nNegative Class Performance (Most Relevant):")
        print(f"   Precision: {metrics['per_class']['precision']['negative']:.4f}")
        print(f"   Recall:    {metrics['per_class']['recall']['negative']:.4f}")
        print(f"   F1-Score:  {metrics['per_class']['f1']['negative']:.4f}")
    
    print(f"\nOutput Files:")
    print(f"   - {Config.OUTPUT_DIR}/{Config.METRICS_REPORT}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.CONFUSION_MATRIX_IMG}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.METRICS_VISUALIZATION}")
    
    print("\n" + "="*70)
    print("Task 1 Collection analysis completed successfully!")
    print("="*70 + "\n")


def main():
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("TASK 1 COLLECTION TEMPLATE DATASET METRICS")
    print("MODEL: ROBERTA-TWITTER")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {Config.INPUT_CSV}")
    print("="*70)
    
    try:
        df = load_dataset(Config.INPUT_CSV)
        metrics = evaluate_model(df)
        create_visualizations(metrics, metrics['labels'])
        save_metrics_report(df, metrics, metrics['labels'])
        display_summary(df, metrics, start_time)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print(f"\nTip: Make sure {Config.INPUT_CSV} exists and has the required columns!")
        raise


if __name__ == "__main__":
    main()
