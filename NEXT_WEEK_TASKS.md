# Next Week's Tasks

## Overview
This document outlines the tasks for the upcoming week, focusing on expanding the Instagram synthetic dataset, running sentiment analysis with RoBERTa, and training a neural network model for account classification using the LIMFAAD dataset.

---

## Task 1: Expand Instagram Synthetic Dataset with Positive Comments

### Objective
Add 150 positive comments to the existing Instagram synthetic dataset to enhance the dataset size and improve model training diversity.

### Steps
1. **Data Collection**
   - Identify and collect 150 positive comments from Instagram posts
   - Ensure comments are authentic and representative of positive sentiment
   - Verify comment quality and relevance

2. **Data Integration**
   - Add the 150 positive comments to the existing synthetic dataset
   - Maintain consistent data format with existing dataset structure
   - Update dataset file: `InstagramPosts_Base.csv` or create new synthetic dataset file

3. **Data Validation**
   - Verify sentiment labels are correctly assigned (positive)
   - Check for data consistency and format compliance
   - Ensure no duplicates or invalid entries

### Deliverables
- [ ] 150 positive comments collected and verified
- [ ] Updated synthetic dataset file with 150 new positive comments
- [ ] Dataset validation report

### Expected Output Files
- Updated `InstagramPosts_Base.csv` or new synthetic dataset file
- `outputs/instagram_dataset_expansion_report.txt` - Expansion summary

---

## Task 2: Run RoBERTa Model on Expanded Instagram Dataset

### Objective
Execute sentiment analysis on the expanded Instagram synthetic dataset using the RoBERTa model to identify negative comments.

### Important Constraints
- **Process only comments** - Do NOT include user details (profile information, metadata, etc.)
- Use the same RoBERTa model configuration as previous phases
- Extract negative comments for downstream processing

### Steps
1. **Data Preparation**
   - Load the expanded Instagram synthetic dataset
   - Extract only the comment text column (exclude user details)
   - Apply text preprocessing (using `preprocess.py` functions)

2. **Model Execution**
   - Load RoBERTa model: `cardiffnlp/twitter-roberta-base-sentiment`
   - Run sentiment classification on comment text only
   - Generate predictions with confidence scores

3. **Negative Comment Extraction**
   - Filter predictions to extract all comments classified as "negative"
   - Save negative comments to a separate file for Task 3

### Configuration
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Input**: Expanded Instagram synthetic dataset (comments only)
- **Batch Size**: 16 (default)
- **Max Length**: 128 (default)

### Deliverables
- [ ] RoBERTa sentiment analysis completed on Instagram dataset
- [ ] Negative comments extracted and saved
- [ ] Sentiment classification results with metrics
- [ ] Confusion matrix and visualization

### Expected Output Files
- `outputs/instagram_roberta_sentiment_results.csv` - Full sentiment predictions
- `outputs/instagram_roberta_negative_comments.csv` - Extracted negative comments
- `outputs/instagram_roberta_confusion_matrix.png` - Confusion matrix visualization
- `outputs/instagram_roberta_metrics_report.txt` - Performance metrics

### Script Reference
- Adapt `roberta_imbalanced_main.py` or `roberta_main.py` for Instagram dataset
- Ensure only comment text is processed (no user metadata)

---

## Task 3: Train Neural Network Model on LIMFAAD Dataset

### Objective
Train a neural network model using the LIMFAAD dataset to classify accounts into categories: spam, bot, fake, and real.

### Steps
1. **Dataset Preparation**
   - Load the LIMFAAD dataset
   - Understand dataset structure and features
   - Perform data preprocessing and feature engineering
   - Split dataset into training and validation sets

2. **Model Architecture**
   - Design neural network architecture suitable for account classification
   - Define input features (may include text features, profile metadata, behavioral patterns)
   - Configure output layer for 4-class classification (spam, bot, fake, real)

3. **Model Training**
   - Train the neural network on LIMFAAD dataset
   - Implement appropriate loss function (e.g., categorical cross-entropy)
   - Monitor training metrics (accuracy, loss, F1-score)
   - Apply early stopping and validation checks

4. **Model Evaluation**
   - Evaluate trained model on validation set
   - Generate classification metrics (accuracy, precision, recall, F1-score per class)
   - Create confusion matrix for 4-class classification
   - Save trained model for inference

### Classification Categories
1. **Spam** - Accounts used primarily for spam activities
2. **Bot** - Automated bot accounts
3. **Fake** - Accounts with suspicious or fake characteristics
4. **Real** - Authentic user accounts

### Deliverables
- [ ] LIMFAAD dataset loaded and preprocessed
- [ ] Neural network model architecture defined
- [ ] Model trained on LIMFAAD dataset
- [ ] Training metrics and validation results
- [ ] Trained model saved for inference
- [ ] Model evaluation report

### Expected Output Files
- `outputs/limfaad_training_metrics.csv` - Training history
- `outputs/limfaad_model_validation_results.csv` - Validation predictions
- `outputs/limfaad_model_confusion_matrix.png` - Confusion matrix
- `outputs/limfaad_model_report.txt` - Detailed evaluation report
- `models/limfaad_trained_model.pth` or `.h5` - Saved trained model

### Implementation Notes
- Use appropriate deep learning framework (PyTorch or TensorFlow/Keras)
- Consider feature engineering for account classification
- Implement proper data validation and preprocessing
- Save model checkpoints during training

---

## Task 4: Classify 150 Synthetic Instagram Dataset with Trained Neural Network

### Objective
Use the trained neural network model (from Task 3) to classify the 150 synthetic Instagram dataset into spam, bot, fake, and real categories.

### Important Note
- **Can include user details** - Unlike Task 2, this classification can use both comment text AND user profile details/metadata
- Use the complete dataset with all available features

### Steps
1. **Data Preparation**
   - Load the 150 synthetic Instagram dataset
   - Include both comment text and user details (followers, following, posts, bio, profile picture, etc.)
   - Apply same preprocessing as used during LIMFAAD model training
   - Feature engineering to match LIMFAAD dataset format

2. **Model Inference**
   - Load the trained neural network model from Task 3
   - Run inference on the 150 synthetic Instagram dataset
   - Generate predictions for each account (spam, bot, fake, real)
   - Calculate confidence scores for each prediction

3. **Results Analysis**
   - Analyze classification distribution across 4 categories
   - Identify patterns in account classifications
   - Compare predictions with expected classifications (if ground truth available)
   - Generate comprehensive classification report

### Input Data
- **150 Synthetic Instagram Dataset** with:
  - Comment/Post text
  - User profile details (followers, following, posts, bio, profile picture, etc.)
  - Any other metadata available

### Deliverables
- [ ] 150 synthetic Instagram dataset classified into 4 categories
- [ ] Classification results with confidence scores
- [ ] Category distribution analysis
- [ ] Classification report and metrics
- [ ] Visualization of classification results

### Expected Output Files
- `outputs/instagram_150_classification_results.csv` - Full classification results
- `outputs/instagram_150_category_distribution.png` - Category distribution chart
- `outputs/instagram_150_classification_matrix.png` - Classification confusion matrix
- `outputs/instagram_150_classification_report.txt` - Detailed classification report
- `outputs/instagram_150_confidence_distribution.png` - Confidence scores visualization

### Analysis Points
- Distribution of accounts across spam, bot, fake, and real categories
- Confidence levels of predictions
- Patterns in account characteristics that lead to specific classifications
- Comparison with sentiment analysis results from Task 2 (if applicable)

---

## Summary of Workflow

```
1. Collect 150 positive comments
   ↓
2. Add to Instagram synthetic dataset
   ↓
3. Run RoBERTa on comments only → Extract negative comments
   ↓
4. Train neural network on LIMFAAD dataset → Save model
   ↓
5. Classify 150 synthetic dataset (with user details) using trained model
   ↓
6. Generate comprehensive results and analysis
```

---

## Key Files and Scripts

### Input Files
- `InstagramPosts_Base.csv` - Base Instagram dataset (to be expanded)
- LIMFAAD dataset file (location to be confirmed)

### Scripts to Create/Adapt
- `instagram_roberta_analysis.py` - RoBERTa sentiment analysis for Instagram (adapt from `roberta_imbalanced_main.py`)
- `train_limfaad_model.py` - Neural network training on LIMFAAD dataset (new script)
- `classify_instagram_150.py` - Classification of 150 synthetic dataset (new script)

### Output Directory
- All results saved to `outputs/` directory
- Trained models saved to `models/` directory (create if needed)

---

## Notes and Considerations

1. **Data Consistency**: Ensure the 150 positive comments match the format and style of existing Instagram dataset
2. **Model Compatibility**: Verify that features from Instagram dataset align with LIMFAAD dataset features for proper model inference
3. **Feature Engineering**: May need to create derived features (ratios, patterns) similar to LIMFAAD dataset
4. **Evaluation Metrics**: Use appropriate metrics for 4-class classification (macro/micro F1, per-class metrics)
5. **Documentation**: Document any assumptions, preprocessing steps, and model configurations

---

## Timeline Estimate

- **Task 1**: 1-2 days (data collection and integration)
- **Task 2**: 1 day (RoBERTa execution and negative comment extraction)
- **Task 3**: 2-3 days (LIMFAAD dataset preparation, model training, and evaluation)
- **Task 4**: 1 day (classification and results analysis)

**Total Estimated Time**: 5-7 days

---

**Last Updated**: 2025
**Status**: Planning Phase
