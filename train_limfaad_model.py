"""
Task 3: Train XGBoost Model on LIMFAAD Dataset
Account Classification: Bot, Scam, Real, Spam

Model: XGBoost Classifier
Input: LIMFADD.csv (LIMFAAD dataset)
Output: Trained model for 4-class account classification
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TASK 3: TRAIN XGBOOST MODEL ON LIMFAAD DATASET")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Model: XGBoost Classifier")
print("Dataset: LIMFADD.csv")
print("="*80)

# Configuration
CONFIG = {
    'input_file': 'LIMFADD.csv',
    'output_dir': 'task3_limfaad/outputs',
    'models_dir': 'task3_limfaad/models',
    'test_size': 0.2,
    'random_state': 42,
    # Label naming alignment:
    # LIMFADD.csv uses: Bot, Scam, Real, Spam
    # Project wants:    Bot, Scam, Real, Spam
    # Keep original labels for the project.
    'map_scam_to_fake': False,
    'xgb_params': {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Create directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)

# Step 1: Load and Explore Dataset
print("\n" + "="*80)
print("STEP 1: LOADING LIMFAAD DATASET")
print("="*80)

print(f"📂 Reading: {CONFIG['input_file']}")
df = pd.read_csv(CONFIG['input_file'])

print(f"✅ Loaded {len(df)} samples")
print(f"📊 Features: {len(df.columns) - 1}")
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

print(f"\n📈 Class distribution:")
print(df['Labels'].value_counts().sort_index())

# Step 2: Data Preprocessing
print("\n" + "="*80)
print("STEP 2: DATA PREPROCESSING")
print("="*80)

# Create a copy for processing
df_processed = df.copy()

# Optional label mapping (only renames for consistency)
if CONFIG.get('map_scam_to_fake', True):
    print("\n🔄 Mapping labels: Scam → Fake (naming alignment)")
    df_processed['Labels'] = df_processed['Labels'].replace('Scam', 'Fake')
else:
    print("\nℹ️ Keeping original dataset labels (Bot/Scam/Real/Spam)")

print("✅ Class distribution used for training:")
print(df_processed['Labels'].value_counts().sort_index())

# Handle #DIV/0! errors in ratio columns
print("\n🔧 Fixing #DIV/0! errors in ratio columns...")

# Convert ratio columns to numeric, replacing #DIV/0! with NaN
df_processed['Following/Followers'] = pd.to_numeric(
    df_processed['Following/Followers'], errors='coerce'
)
df_processed['Posts/Followers'] = pd.to_numeric(
    df_processed['Posts/Followers'], errors='coerce'
)

# Fill NaN values (from #DIV/0!) with 0 or appropriate value
# For Following/Followers: if Followers=0, ratio should be high (set to max or large value)
# For Posts/Followers: if Followers=0, ratio should be 0
df_processed.loc[
    (df_processed['Following/Followers'].isna()) & (df_processed['Followers'] == 0),
    'Following/Followers'
] = df_processed['Following'].max()  # Large value when no followers

df_processed.loc[
    (df_processed['Posts/Followers'].isna()) & (df_processed['Followers'] == 0),
    'Posts/Followers'
] = 0  # No posts per follower when no followers

# Fill any remaining NaN with median
df_processed['Following/Followers'].fillna(
    df_processed['Following/Followers'].median(), inplace=True
)
df_processed['Posts/Followers'].fillna(
    df_processed['Posts/Followers'].median(), inplace=True
)

print(f"✅ Fixed {df['Following/Followers'].astype(str).str.contains('#DIV').sum()} #DIV/0! errors")

# Normalize Bio column (Yes/yes → 1, N → 0)
print("\n🔧 Encoding categorical features...")
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

print("✅ Encoded categorical features (Bio, Profile Picture, External Link, Threads)")

# Prepare features and labels
feature_columns = [
    'Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers',
    'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads'
]

X = df_processed[feature_columns].copy()
y = df_processed['Labels'].copy()

print(f"\n✅ Features prepared: {len(feature_columns)} features")
print(f"   Feature names: {feature_columns}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"\n✅ Labels encoded:")
for i, class_name in enumerate(class_names):
    print(f"   {i}: {class_name}")

# Step 3: Split Dataset
print("\n" + "="*80)
print("STEP 3: SPLITTING DATASET")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y_encoded
)

print(f"✅ Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✅ Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n📊 Training set class distribution:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_dist.items():
    print(f"   {class_names[idx]}: {count} ({count/len(y_train)*100:.1f}%)")

# Step 4: Train XGBoost Model
print("\n" + "="*80)
print("STEP 4: TRAINING XGBOOST MODEL")
print("="*80)

print("🤖 Model Configuration:")
for key, value in CONFIG['xgb_params'].items():
    print(f"   {key}: {value}")

print("\n⏳ Training XGBoost classifier...")
print("   (This may take a few minutes)")

model = xgb.XGBClassifier(**CONFIG['xgb_params'])

# Train with early stopping on validation set
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=CONFIG['random_state'],
    stratify=y_train
)

model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("✅ Model training completed!")

# Step 5: Model Evaluation
print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION")
print("="*80)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Probabilities
y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, average=None, labels=range(len(class_names))
)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='macro'
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='weighted'
)

print(f"\n📊 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n📈 Per-Class Metrics (Test Set):")
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 60)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

print(f"\n📊 Overall Metrics (Test Set):")
print(f"   Macro Precision: {macro_precision:.4f}")
print(f"   Macro Recall: {macro_recall:.4f}")
print(f"   Macro F1-Score: {macro_f1:.4f}")
print(f"   Weighted Precision: {weighted_precision:.4f}")
print(f"   Weighted Recall: {weighted_recall:.4f}")
print(f"   Weighted F1-Score: {weighted_f1:.4f}")

# Step 6: Feature Importance
print("\n" + "="*80)
print("STEP 6: FEATURE IMPORTANCE")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<25} {row['importance']:.4f}")

# Step 7: Save Results
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

# Save training metrics
training_metrics = pd.DataFrame({
    'metric': ['train_accuracy', 'test_accuracy', 'macro_precision', 'macro_recall', 
               'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1'],
    'value': [train_accuracy, test_accuracy, macro_precision, macro_recall,
              macro_f1, weighted_precision, weighted_recall, weighted_f1]
})
training_metrics_file = f"{CONFIG['output_dir']}/limfaad_training_metrics.csv"
training_metrics.to_csv(training_metrics_file, index=False)
print(f"✅ Saved: {training_metrics_file}")

# Save validation results
validation_results = pd.DataFrame({
    'true_label': [class_names[y] for y in y_test],
    'predicted_label': [class_names[y] for y in y_test_pred],
    'confidence': [max(proba) for proba in y_test_proba]
})
for i, class_name in enumerate(class_names):
    validation_results[f'prob_{class_name.lower()}'] = y_test_proba[:, i]

validation_results_file = f"{CONFIG['output_dir']}/limfaad_model_validation_results.csv"
validation_results.to_csv(validation_results_file, index=False)
print(f"✅ Saved: {validation_results_file}")

# Save feature importance
feature_importance_file = f"{CONFIG['output_dir']}/limfaad_feature_importance.csv"
feature_importance.to_csv(feature_importance_file, index=False)
print(f"✅ Saved: {feature_importance_file}")

# Save detailed classification report
report_file = f"{CONFIG['output_dir']}/limfaad_model_report.txt"
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("LIMFAAD MODEL EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: XGBoost Classifier\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {CONFIG['input_file']}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("MODEL CONFIGURATION\n")
    f.write("="*80 + "\n")
    for key, value in CONFIG['xgb_params'].items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
    
    f.write("Per-Class Metrics:\n")
    f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
    f.write("-" * 60 + "\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}\n")
    
    f.write("\nOverall Metrics:\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1-Score: {macro_f1:.4f}\n")
    f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
    f.write(f"Weighted Recall: {weighted_recall:.4f}\n")
    f.write(f"Weighted F1-Score: {weighted_f1:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_test, y_test_pred, target_names=class_names))
    
    f.write("\n" + "="*80 + "\n")
    f.write("FEATURE IMPORTANCE\n")
    f.write("="*80 + "\n\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['feature']:<25} {row['importance']:.4f}\n")

print(f"✅ Saved: {report_file}")

# Step 8: Visualizations
print("\n" + "="*80)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*80)

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
confusion_matrix_file = f"{CONFIG['output_dir']}/limfaad_model_confusion_matrix.png"
plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {confusion_matrix_file}")

# Feature Importance Plot
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 10 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
feature_importance_plot = f"{CONFIG['output_dir']}/limfaad_feature_importance_plot.png"
plt.savefig(feature_importance_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {feature_importance_plot}")

# Confidence Distribution
plt.figure(figsize=(10, 6))
confidences = [max(proba) for proba in y_test_proba]
plt.hist(confidences, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Confidence Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Confidence Scores (Test Set)', fontsize=14, fontweight='bold')
plt.axvline(np.mean(confidences), color='red', linestyle='--', 
            label=f'Mean: {np.mean(confidences):.3f}')
plt.legend()
plt.tight_layout()
confidence_dist_file = f"{CONFIG['output_dir']}/limfaad_confidence_distribution.png"
plt.savefig(confidence_dist_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {confidence_dist_file}")

# Step 9: Save Model and Label Encoder
print("\n" + "="*80)
print("STEP 9: SAVING MODEL")
print("="*80)

# Save XGBoost model
model_file = f"{CONFIG['models_dir']}/limfaad_xgboost_model.json"
model.save_model(model_file)
print(f"✅ Saved: {model_file}")

# Save label encoder
encoder_file = f"{CONFIG['models_dir']}/limfaad_label_encoder.pkl"
with open(encoder_file, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✅ Saved: {encoder_file}")

# Save feature names for inference
feature_info = {
    'feature_columns': feature_columns,
    'class_names': class_names.tolist()
}
feature_info_file = f"{CONFIG['models_dir']}/limfaad_feature_info.pkl"
with open(feature_info_file, 'wb') as f:
    pickle.dump(feature_info, f)
print(f"✅ Saved: {feature_info_file}")

# Final Summary
print("\n" + "="*80)
print("TASK 3 COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n📊 Final Results:")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Macro F1-Score: {macro_f1:.4f}")
print(f"   Weighted F1-Score: {weighted_f1:.4f}")

print(f"\n📁 Output Files:")
print(f"   Model: {model_file}")
print(f"   Label Encoder: {encoder_file}")
print(f"   Feature Info: {feature_info_file}")
print(f"   Training Metrics: {training_metrics_file}")
print(f"   Validation Results: {validation_results_file}")
print(f"   Evaluation Report: {report_file}")
print(f"   Confusion Matrix: {confusion_matrix_file}")
print(f"   Feature Importance Plot: {feature_importance_plot}")
print(f"   Confidence Distribution: {confidence_dist_file}")

print(f"\n✅ Model is ready for Task 4 (Instagram dataset classification)")
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
