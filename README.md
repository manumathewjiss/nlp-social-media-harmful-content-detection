# NLP Sentiment Analysis Project

A comprehensive three-phase sentiment analysis and text classification project using state-of-the-art transformer models. This project implements sentiment classification (Phase 1), advanced negative comment categorization (Phase 2), and account authenticity detection (Phase 3) with extensive model evaluation, validation, and visualization capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Phase 1: Sentiment Analysis](#phase-1-sentiment-analysis)
- [Phase 2: Negative Comment Classification](#phase-2-negative-comment-classification)
- [Phase 3: Account Authenticity Detection](#phase-3-account-authenticity-detection)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Output Files](#output-files)
- [Validation](#validation)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [License](#license)

## 🎯 Overview

This project implements a three-phase approach to text sentiment analysis, toxicity classification, and account authenticity detection:

- **Phase 1**: Sentiment classification into three categories (Positive, Negative, Neutral) using RoBERTa models
- **Phase 2**: Advanced classification of negative comments into 7 specific toxicity categories using BART zero-shot classification
- **Phase 3**: Account authenticity detection to classify accounts as Fake, Real, Bot, or Spam based on user profile data and toxicity classifications from Phase 2

The project includes comprehensive model comparison, validation scripts, and detailed visualization of results including confusion matrices, accuracy reports, and distribution charts.

## ✨ Features

- **Multi-Model Support**: Comparison between BERTweet and RoBERTa models
- **Dataset Variants**: Support for both balanced and imbalanced datasets
- **Three-Phase Pipeline**: Sentiment analysis → Toxicity categorization → Account authenticity detection
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, classification reports
- **Visualization**: Automated generation of charts and graphs for model performance
- **Validation Scripts**: Dedicated validation for both phases with manual test sets
- **Model Caching**: Efficient model storage and reuse via local cache
- **Text Preprocessing**: Automated cleaning and normalization of input text

## 📁 Project Structure

```
Tap_Project/
├── main.py                          # BERTweet sentiment analysis (balanced dataset)
├── roberta_main.py                  # RoBERTa sentiment analysis (balanced dataset)
├── roberta_imbalanced_main.py       # RoBERTa sentiment analysis (imbalanced dataset) ⭐
├── phase2_main.py                   # Phase 2: Negative comment classification
├── phase3_main.py                   # Phase 3: Account authenticity detection (planned)
├── validate_phase1.py              # Phase 1 validation script
├── phase2_validation.py            # Phase 2 validation script
├── compare_all_models.py           # Model comparison utility
├── create_balanced_dataset.py      # Dataset balancing utility
├── preprocess.py                   # Text preprocessing functions
├── requirements.txt                # Python dependencies
├── outputs/                        # Generated results and visualizations
│   ├── phase1_*.csv               # Phase 1 results
│   ├── phase1_*.png               # Phase 1 visualizations
│   ├── phase2_*.csv               # Phase 2 results
│   ├── phase2_*.png               # Phase 2 visualizations
│   ├── phase3_*.csv               # Phase 3 results
│   ├── phase3_*.png               # Phase 3 visualizations
│   └── model_comparison_*.png     # Model comparison charts
├── model_cache/                    # Cached transformer models (gitignored)
├── scripts/                        # Utility scripts
└── YoutubeCommentsDataSet.csv      # Main dataset (imbalanced)
```

## 🔬 Phase 1: Sentiment Analysis

Phase 1 performs three-class sentiment classification (Positive, Negative, Neutral) on text data.

### Models Evaluated

1. **BERTweet** (`finiteautomata/bertweet-base-sentiment-analysis`)
   - Balanced dataset
   - Imbalanced dataset

2. **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment`) ⭐ **Selected Model**
   - Balanced dataset
   - Imbalanced dataset
   - **Best Performance**: 90.23% accuracy on imbalanced dataset

### Key Features

- Batch processing for efficient inference
- Text preprocessing (URL removal, emoji handling, normalization)
- Confidence score calculation
- Comprehensive metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Export of predictions with confidence scores

### Usage

```bash
# Run RoBERTa on imbalanced dataset (recommended)
python roberta_imbalanced_main.py

# Run RoBERTa on balanced dataset
python roberta_main.py

# Run BERTweet on balanced dataset
python main.py
```

### Output

- CSV file with predictions and confidence scores
- Confusion matrix visualization (PNG)
- Accuracy report (TXT)
- Negative comments extracted for Phase 2

## 🎯 Phase 2: Negative Comment Classification

Phase 2 takes negative comments identified in Phase 1 and classifies them into 7 specific categories using BART zero-shot classification.

### Classification Categories

1. **Harassment or Hate Speech**
2. **Spam**
3. **Inappropriate Content**
4. **Toxicity**
5. **Aggressive Behavior**
6. **Misinformation**
7. **Other Negative**

### Model

- **BART Large MNLI** (`facebook/bart-large-mnli`)
  - Zero-shot classification capability
  - No fine-tuning required
  - Handles multiple categories simultaneously

### Key Features

- Zero-shot classification (no training required)
- Multi-class categorization
- Category distribution analysis
- Classification matrix visualization
- Confidence distribution charts

### Usage

```bash
# Run Phase 2 classification
python phase2_main.py
```

**Input**: `outputs/phase2_input_negative_comments_roberta_imbalanced.csv` (2,337 negative comments)

### Output

- CSV file with category classifications
- Category distribution chart
- Classification confusion matrix
- Classification report with metrics
- Confidence distribution visualization

## 🤖 Phase 3: Account Authenticity Detection

Phase 3 analyzes user accounts to determine their authenticity by combining toxicity classifications from Phase 2 with user profile metadata to classify accounts as Fake, Real, Bot, or Spam.

### Classification Categories

1. **Fake** - Accounts with suspicious profile characteristics
2. **Real** - Authentic user accounts
3. **Bot** - Automated bot accounts
4. **Spam** - Accounts primarily used for spam activities

### Input Data

- **Phase 2 Results**: Toxicity classifications (harassment, spam, toxicity, etc.)
- **User Profile Data**: From `task1_collection_template.csv` including:
  - Username and Channel information
  - Account creation date
  - Subscriber count
  - Video count
  - Profile picture status
  - Username patterns
  - Channel metadata

### Key Features

- Multi-feature analysis combining text toxicity and profile metadata
- Account classification based on behavioral and profile patterns
- Integration of Phase 2 toxicity results with user profile data
- Detection of suspicious account patterns

### Usage

```bash
# Run Phase 3 account authenticity detection
python phase3_main.py
```

**Input**: 
- `outputs/phase2_classification_results.csv` (Phase 2 toxicity classifications)
- `task1_collection_template.csv` (User profile data)

### Output

- CSV file with account classifications (Fake/Real/Bot/Spam)
- Account type distribution chart
- Classification matrix visualization
- Detailed report with account authenticity metrics
- Profile feature analysis

## 🤖 Models Used

### Phase 1 Models

| Model | Dataset | Accuracy | Status |
|-------|---------|----------|--------|
| RoBERTa | Imbalanced | **90.23%** | ✅ Selected |
| RoBERTa | Balanced | ~88-89% | Evaluated |
| BERTweet | Balanced | ~85-87% | Evaluated |
| BERTweet | Imbalanced | ~83-85% | Evaluated |

### Phase 2 Model

- **BART Large MNLI**: Zero-shot classification model for multi-class categorization
- No accuracy metrics available (zero-shot, no labeled training data)

### Phase 3 Model

- **Account Classification**: Feature-based classification using profile metadata and toxicity scores
- Combines rule-based heuristics with machine learning approaches
- Uses profile features: subscriber count, video count, account age, username patterns, etc.

### LIMFAAD Account Classification (XGBoost vs BERT)

Account-type classification (Bot, Scam, Real, Spam) is trained on the **LIMFAAD** dataset and applied to Instagram negative-comment accounts:

- **XGBoost** (`train_limfaad_model.py`): Trains on tabular profile features (followers, following, ratios, bio, profile picture, etc.). Same pipeline as above.
- **BERT** (`train_limfaad_bert.py`): Same LIMFAAD data and train/test split; each row is converted to a short text description and BERT is fine-tuned for 4-class classification. Enables direct comparison with XGBoost on the same task.

**Workflow:**

1. Train XGBoost on LIMFAAD: `python train_limfaad_model.py`
2. Train BERT on LIMFAAD: `python train_limfaad_bert.py`
3. Compare both on LIMFAAD test set: `python compare_limfaad_models.py`
4. Classify Instagram negative-comment accounts with both models and compare: `python classify_negative_comments_compare.py`

Outputs: `task3_limfaad/outputs/` (metrics, reports), `task4_classification/` (Instagram predictions and XGBoost vs BERT comparison).

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/manumathewjiss/nlp-sentiment-project.git
   cd nlp-sentiment-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models** (automatically on first run)
   - Models will be cached in `model_cache/` directory
   - First run may take time to download models (~500MB-1GB)

### Dependencies

See `requirements.txt` for complete list. Key dependencies:

- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.30.0` - Hugging Face transformers library
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning metrics
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `tqdm>=4.65.0` - Progress bars

## 📖 Usage

### Quick Start

1. **Run Phase 1 (Sentiment Analysis)**
   ```bash
   python roberta_imbalanced_main.py
   ```
   This will:
   - Load the dataset
   - Run sentiment classification
   - Generate results and visualizations
   - Extract negative comments for Phase 2

2. **Run Phase 2 (Negative Comment Classification)**
   ```bash
   python phase2_main.py
   ```
   This will:
   - Load negative comments from Phase 1
   - Classify into 7 categories
   - Generate classification results and visualizations

3. **Run Phase 3 (Account Authenticity Detection)**
   ```bash
   python phase3_main.py
   ```
   This will:
   - Load Phase 2 toxicity classifications
   - Load user profile data
   - Analyze account features and patterns
   - Classify accounts as Fake/Real/Bot/Spam
   - Generate account authenticity reports

### Validation

**Phase 1 Validation**
```bash
python validate_phase1.py
```
Validates the model on 100 manually collected negative comments.

**Phase 2 Validation**
```bash
python phase2_validation.py
```
Validates Phase 2 classification on manually labeled data.

### Model Comparison

Compare all model variants:
```bash
python compare_all_models.py
```
Generates comprehensive comparison report and visualizations.

### LIMFAAD: XGBoost vs BERT (account classification)

Train both models on LIMFAAD and compare on Instagram negative-comment accounts:

```bash
# 1. Train XGBoost on LIMFAAD (tabular features)
python train_limfaad_model.py

# 2. Train BERT on LIMFAAD (tabular rows as text)
python train_limfaad_bert.py

# 3. Compare XGBoost vs BERT on the same LIMFAAD test set
python compare_limfaad_models.py

# 4. Classify Instagram negative-comment accounts with both models and compare
python classify_negative_comments_compare.py
```

### Dataset Balancing

Create a balanced dataset from imbalanced data:
```bash
python create_balanced_dataset.py
```

## 📊 Results & Performance

### Phase 1 Results

**RoBERTa (Imbalanced Dataset)** - Selected Model
- **Accuracy**: 90.23%
- **Dataset Size**: Full imbalanced dataset
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Negative Comments Extracted**: 2,337

### Phase 2 Results

- **Input**: 2,337 negative comments from Phase 1
- **Categories**: 7 classification categories
- **Model**: BART Large MNLI (zero-shot)
- **Output**: Categorized negative comments with confidence scores

### Output Files Location

All results are saved in the `outputs/` directory:
- CSV files with predictions
- PNG files with visualizations
- TXT files with detailed reports

## 📄 Output Files

### Phase 1 Outputs

- `phase1_sentiment_results_roberta_imbalanced.csv` - Full predictions
- `phase1_confusion_matrix_roberta_imbalanced.png` - Confusion matrix
- `phase2_input_negative_comments_roberta_imbalanced.csv` - Negative comments for Phase 2

### Phase 2 Outputs

- `phase2_classification_results.csv` - Category classifications
- `phase2_classification_matrix.png` - Classification matrix
- `phase2_category_distribution.png` - Category distribution
- `phase2_classification_report.txt` - Detailed metrics
- `phase2_confidence_distribution.png` - Confidence scores

### Phase 3 Outputs

- `phase3_account_classification_results.csv` - Account authenticity classifications
- `phase3_account_type_distribution.png` - Account type distribution
- `phase3_classification_matrix.png` - Classification matrix
- `phase3_account_report.txt` - Detailed account analysis
- `phase3_profile_analysis.png` - Profile feature analysis

### Validation Outputs

- `phase1_validation_results.csv` - Phase 1 validation results
- `phase2_validation_classification_results.csv` - Phase 2 validation results
- Corresponding visualization and report files

## ✅ Validation

The project includes comprehensive validation scripts:

1. **Phase 1 Validation**: Tests sentiment classification on 100 manually collected negative comments
2. **Phase 2 Validation**: Tests category classification on manually labeled data
3. **Phase 3 Validation**: Tests account authenticity detection on manually verified accounts

Validation scripts generate:
- Accuracy metrics
- Confusion matrices
- Prediction distributions
- Confidence score analysis
- Detailed validation reports

## 🔧 Configuration

Key configuration parameters can be modified in each script's `Config` class:

- `BATCH_SIZE`: Batch size for inference (default: 16)
- `MAX_LENGTH`: Maximum sequence length (default: 128)
- `MODEL_NAME`: Transformer model to use
- `CACHE_DIR`: Directory for model caching
- `OUTPUT_DIR`: Directory for output files

## 📝 Text Preprocessing

The `preprocess.py` module handles:
- Lowercasing
- URL removal
- Mention/hashtag removal
- Special character normalization
- Whitespace cleanup

## 🗂️ File Structure

### Main Scripts

- `main.py` - BERTweet sentiment analysis (balanced)
- `roberta_main.py` - RoBERTa sentiment analysis (balanced)
- `roberta_imbalanced_main.py` - RoBERTa sentiment analysis (imbalanced) ⭐
- `phase2_main.py` - Phase 2 negative comment classification
- `phase3_main.py` - Phase 3 account authenticity detection
- `train_limfaad_model.py` - Train XGBoost on LIMFAAD for account classification (Bot/Scam/Real/Spam)
- `train_limfaad_bert.py` - Train BERT on LIMFAAD (tabular-as-text) for the same 4-class task
- `classify_negative_comments_xgboost.py` - Classify Instagram negative-comment accounts with XGBoost
- `classify_negative_comments_compare.py` - Classify with both XGBoost and BERT and compare results
- `compare_limfaad_models.py` - Compare XGBoost vs BERT metrics on LIMFAAD test set
- `validate_phase1.py` - Phase 1 validation
- `phase2_validation.py` - Phase 2 validation
- `compare_all_models.py` - Model comparison utility (Phase 1 sentiment models)

### Data Files

- `YoutubeCommentsDataSet.csv` - Main imbalanced dataset
- `YoutubeCommentsDataSet_Balanced.csv` - Balanced dataset variant
- `task1_collection_template.csv` - Manual validation data with user profile metadata

### Utilities

- `preprocess.py` - Text preprocessing functions
- `create_balanced_dataset.py` - Dataset balancing utility
- `scripts/calculate_profile_score.py` - Additional utility scripts

## 🔍 Notes

- **Model Cache**: Large model files are cached locally in `model_cache/` (gitignored)
- **Dataset**: The imbalanced dataset is used as the primary dataset for Phase 1
- **Selected Model**: RoBERTa on imbalanced dataset achieved best performance (90.23% accuracy)
- **Phase 2 Input**: Uses negative comments from Phase 1 RoBERTa imbalanced results
- **Zero-Shot Learning**: Phase 2 uses zero-shot classification (no training required)
- **Phase 3 Input**: Combines Phase 2 toxicity classifications with user profile data from `task1_collection_template.csv`
- **Account Detection**: Phase 3 analyzes profile features (subscriber count, video count, account age, username patterns) along with toxicity scores to classify account authenticity

## 📈 Future Enhancements

Potential improvements:
- Fine-tuning Phase 2 model on labeled data
- Additional model architectures
- Real-time inference API
- Web interface for predictions
- Additional preprocessing options
- Hyperparameter tuning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Manu Mathew Jiss**

## 🙏 Acknowledgments

- Hugging Face for transformer models and libraries
- Cardiff NLP for the RoBERTa sentiment model
- Facebook AI for BART model
- All contributors to the open-source ML community

---

**Repository**: [https://github.com/manumathewjiss/nlp-sentiment-project](https://github.com/manumathewjiss/nlp-sentiment-project)

**Last Updated**: 2025
