"""
Create comprehensive metrics visualization for LIMFAAD XGBoost Model
Shows: Accuracy, Precision, Recall, F1-Score (overall and per-class)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING LIMFAAD MODEL METRICS VISUALIZATION")
print("="*80)

# Load validation results
validation_file = 'task3_limfaad/outputs/limfaad_model_validation_results.csv'
print(f"\n📂 Loading validation results: {validation_file}")
df = pd.read_csv(validation_file)

# Load label encoder to get class names
encoder_file = 'task3_limfaad/models/limfaad_label_encoder.pkl'
print(f"📂 Loading label encoder: {encoder_file}")
with open(encoder_file, 'rb') as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_
print(f"✅ Classes: {class_names}")

# Calculate metrics
y_true = df['true_label'].values
y_pred = df['predicted_label'].values

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=class_names, average=None
)

# Macro averages
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print(f"\n📊 Overall Metrics:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Macro Precision: {macro_precision:.4f}")
print(f"   Macro Recall: {macro_recall:.4f}")
print(f"   Macro F1-Score: {macro_f1:.4f}")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Overall Metrics Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
overall_metrics = {
    'Accuracy': accuracy * 100,
    'Precision': macro_precision * 100,
    'Recall': macro_recall * 100,
    'F1-Score': macro_f1 * 100
}
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax1.bar(overall_metrics.keys(), overall_metrics.values(), color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels
for bar, (metric, value) in zip(bars, overall_metrics.items()):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# 2. Per-Class Precision, Recall, F1-Score
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(class_names))
width = 0.25

metrics_data = {
    'Precision': precision * 100,
    'Recall': recall * 100,
    'F1-Score': f1 * 100
}

x_pos = x - width
for i, (metric_name, values) in enumerate(metrics_data.items()):
    bars = ax2.bar(x_pos, values, width, label=metric_name, 
                   color=colors[i], edgecolor='black', linewidth=1, alpha=0.8)
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    x_pos = x_pos + width

ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(class_names)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# 3. Per-Class Precision Comparison
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.barh(class_names, precision * 100, color='#3498db', 
                edgecolor='black', linewidth=1.5, alpha=0.8)
for i, (bar, val) in enumerate(zip(bars, precision * 100)):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
            f' {val:.2f}%',
            ha='left', va='center', fontsize=11, fontweight='bold')

ax3.set_xlabel('Precision (%)', fontsize=12, fontweight='bold')
ax3.set_title('Per-Class Precision', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlim(0, 105)
ax3.grid(axis='x', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# 4. Per-Class F1-Score Comparison
ax4 = fig.add_subplot(gs[1, 1])
bars = ax4.barh(class_names, f1 * 100, color='#f39c12', 
                edgecolor='black', linewidth=1.5, alpha=0.8)
for i, (bar, val) in enumerate(zip(bars, f1 * 100)):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f' {val:.2f}%',
            ha='left', va='center', fontsize=11, fontweight='bold')

ax4.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax4.set_title('Per-Class F1-Score', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlim(0, 105)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.set_axisbelow(True)

# Add main title
fig.suptitle('LIMFAAD XGBoost Model - Comprehensive Performance Metrics\nBot/Scam/Real/Spam Classification', 
              fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save visualization
output_file = 'task3_limfaad/outputs/limfaad_metrics_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved comprehensive metrics visualization: {output_file}")

# Also create a detailed metrics table CSV
metrics_table = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

# Add overall metrics row
overall_row = pd.DataFrame({
    'Class': ['Overall (Macro Avg)'],
    'Precision': [macro_precision],
    'Recall': [macro_recall],
    'F1-Score': [macro_f1],
    'Support': [len(df)]
})

metrics_table = pd.concat([metrics_table, overall_row], ignore_index=True)

# Add accuracy row
accuracy_row = pd.DataFrame({
    'Class': ['Accuracy'],
    'Precision': [accuracy],
    'Recall': [accuracy],
    'F1-Score': [accuracy],
    'Support': [len(df)]
})

metrics_table = pd.concat([metrics_table, accuracy_row], ignore_index=True)

metrics_csv = 'task3_limfaad/outputs/limfaad_detailed_metrics.csv'
metrics_table.to_csv(metrics_csv, index=False)
print(f"✅ Saved detailed metrics table: {metrics_csv}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\n📊 Generated Files:")
print(f"   1. {output_file}")
print(f"   2. {metrics_csv}")
print("\n📈 Metrics Summary:")
print(f"   Overall Accuracy: {accuracy*100:.2f}%")
print(f"   Macro Precision: {macro_precision*100:.2f}%")
print(f"   Macro Recall: {macro_recall*100:.2f}%")
print(f"   Macro F1-Score: {macro_f1*100:.2f}%")
print("\n📋 Per-Class Metrics:")
for i, class_name in enumerate(class_names):
    print(f"   {class_name}:")
    print(f"      Precision: {precision[i]*100:.2f}%")
    print(f"      Recall: {recall[i]*100:.2f}%")
    print(f"      F1-Score: {f1[i]*100:.2f}%")
    print(f"      Support: {int(support[i])}")
print("="*80)
