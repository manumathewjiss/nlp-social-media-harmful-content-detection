"""
LIME (Local Interpretable Model-Agnostic Explanations) for the BERT account classifier.

Loads the fine-tuned BERT model, selects one correctly-classified example per class
(Bot, Real, Scam, Spam), applies LimeTextExplainer, and produces:

  task3_limfaad/outputs/lime/lime_bert_word_importance.png   ← main paper figure (2×2)
  task3_limfaad/outputs/lime/lime_bert_<class>.html          ← interactive HTML (x4)
  task3_limfaad/outputs/lime/lime_bert_weights_summary.csv   ← all word weights

Run:
    pip install lime          (once)
    python lime_bert_explanations.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config – must match train_limfaad_bert.py exactly
# ---------------------------------------------------------------------------
BERT_MODEL_DIR = "task3_limfaad/models/limfaad_bert"
FEATURE_INFO_PATH = "task3_limfaad/models/limfaad_bert_feature_info.pkl"
LABEL_ENC_PATH = "task3_limfaad/models/limfaad_bert_label_encoder.pkl"
LIME_OUTPUT_DIR = "task3_limfaad/outputs/lime"
INPUT_FILE = "LIMFADD.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 128
BATCH_SIZE = 8          # for LIME perturbation batches
N_LIME_SAMPLES = 800    # perturbations per explanation (lower = faster; 800 is good for paper)
TOP_N_WORDS = 12        # words to display per subplot

os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Shared text conversion (mirrors limfaad_bert_utils.py)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers',
    'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads'
]

def row_to_text(row: pd.Series) -> str:
    def _bool_str(val):
        if isinstance(val, str):
            return val.strip().lower() in ('yes', 'y', '1')
        return int(val) != 0

    followers   = int(row.get('Followers', 0))
    following   = int(row.get('Following', 0))
    ff_ratio    = float(row.get('Following/Followers', 0))
    posts       = int(row.get('Posts', 0))
    pf_ratio    = float(row.get('Posts/Followers', 0))
    bio         = 'yes' if _bool_str(row.get('Bio', 0))            else 'no'
    profile_pic = 'yes' if _bool_str(row.get('Profile Picture', 0)) else 'no'
    external    = 'yes' if _bool_str(row.get('External Link', 0))   else 'no'
    mutual      = int(row.get('Mutual Friends', 0))
    threads     = 'yes' if _bool_str(row.get('Threads', 0))        else 'no'

    return (
        f"Account has {followers} followers and {following} following. "
        f"Following to followers ratio is {ff_ratio:.4f}. "
        f"Posts count is {posts}. Posts per follower ratio is {pf_ratio:.4f}. "
        f"Bio: {bio}. Profile picture: {profile_pic}. External link: {external}. "
        f"Mutual friends: {mutual}. Threads: {threads}."
    )

# ---------------------------------------------------------------------------
# Preprocessing (same as train_limfaad_bert.py)
# ---------------------------------------------------------------------------
def load_and_preprocess_limfaad(csv_path: str):
    df = pd.read_csv(csv_path)
    dp = df.copy()

    dp['Following/Followers'] = pd.to_numeric(dp['Following/Followers'], errors='coerce')
    dp['Posts/Followers']     = pd.to_numeric(dp['Posts/Followers'],     errors='coerce')

    dp.loc[(dp['Following/Followers'].isna()) & (dp['Followers'] == 0),
           'Following/Followers'] = dp['Following'].max()
    dp.loc[(dp['Posts/Followers'].isna()) & (dp['Followers'] == 0),
           'Posts/Followers'] = dp['Posts'].max()

    dp['Following/Followers'].fillna(dp['Following/Followers'].median(), inplace=True)
    dp['Posts/Followers'].fillna(dp['Posts/Followers'].median(), inplace=True)

    for col in ['Bio', 'Profile Picture', 'External Link', 'Threads']:
        if col in dp.columns:
            dp[col] = dp[col].map(
                lambda v: 1 if str(v).strip().lower() in ('yes', 'y', '1', '1.0') else 0
            )

    dp['Mutual Friends'] = pd.to_numeric(dp['Mutual Friends'], errors='coerce').fillna(0)

    label_col = 'Account Type' if 'Account Type' in dp.columns else dp.columns[-1]
    dp = dp.dropna(subset=[label_col])

    le = LabelEncoder()
    dp['label'] = le.fit_transform(dp[label_col].str.strip())

    X = dp[FEATURE_COLUMNS].copy()
    y = dp['label'].values
    return X, y, le

# ---------------------------------------------------------------------------
# Load BERT model
# ---------------------------------------------------------------------------
print("Loading BERT model …")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
model.eval()
model.to(DEVICE)

CLASS_NAMES = ['Bot', 'Real', 'Scam', 'Spam']   # alphabetical – matches LabelEncoder
NUM_LABELS  = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# LIME prediction function
# ---------------------------------------------------------------------------
def bert_predict(texts: list) -> np.ndarray:
    """Returns (N, num_labels) probability array for a list of text strings."""
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

# ---------------------------------------------------------------------------
# Build test set (same split as training)
# ---------------------------------------------------------------------------
print("Loading and preprocessing LIMFAAD …")
X, y, le = load_and_preprocess_limfaad(INPUT_FILE)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_test = X_test.reset_index(drop=True)

# Convert test rows to texts
test_texts = [row_to_text(X_test.iloc[i]) for i in range(len(X_test))]

# ---------------------------------------------------------------------------
# Select one correctly-classified example per class
# ---------------------------------------------------------------------------
print("Selecting representative examples …")
probs_all = bert_predict(test_texts)
y_pred    = np.argmax(probs_all, axis=1)
conf_all  = np.max(probs_all, axis=1)

selected = {}          # class_idx → (text, true_label, pred_label, confidence, row_idx)
for cls_idx in range(NUM_LABELS):
    cls_name = CLASS_NAMES[cls_idx]
    # prefer correctly classified with highest confidence
    mask = (y_test == cls_idx) & (y_pred == cls_idx)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        # fall back to any example of this true class
        idxs = np.where(y_test == cls_idx)[0]
    best = idxs[np.argmax(conf_all[idxs])]
    selected[cls_idx] = {
        "text":       test_texts[best],
        "true_label": CLASS_NAMES[y_test[best]],
        "pred_label": CLASS_NAMES[y_pred[best]],
        "confidence": float(conf_all[best]),
        "row_idx":    int(best),
    }
    print(f"  {cls_name}: row {best} | true={CLASS_NAMES[y_test[best]]} "
          f"pred={CLASS_NAMES[y_pred[best]]} conf={conf_all[best]:.3f}")

# ---------------------------------------------------------------------------
# Run LIME
# ---------------------------------------------------------------------------
explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    split_expression=r'\s+',     # split on whitespace (keeps numbers intact)
    bow=True,
    random_state=RANDOM_STATE,
)

explanations = {}
all_weights  = []

for cls_idx, info in selected.items():
    cls_name  = CLASS_NAMES[cls_idx]
    pred_idx  = CLASS_NAMES.index(info["pred_label"])
    print(f"\nRunning LIME for class '{cls_name}' ({N_LIME_SAMPLES} samples) …")

    exp = explainer.explain_instance(
        info["text"],
        bert_predict,
        labels=[pred_idx],
        num_features=TOP_N_WORDS,
        num_samples=N_LIME_SAMPLES,
    )
    explanations[cls_idx] = exp

    # Save HTML
    html_path = os.path.join(LIME_OUTPUT_DIR, f"lime_bert_{cls_name.lower()}.html")
    exp.save_to_file(html_path)
    print(f"  HTML saved → {html_path}")

    # Collect weights
    for word, weight in exp.as_list(label=pred_idx):
        all_weights.append({
            "class": cls_name,
            "word": word,
            "weight": weight,
            "true_label": info["true_label"],
            "pred_label": info["pred_label"],
            "confidence": info["confidence"],
        })

# Save CSV summary
weights_df = pd.DataFrame(all_weights)
csv_path   = os.path.join(LIME_OUTPUT_DIR, "lime_bert_weights_summary.csv")
weights_df.to_csv(csv_path, index=False)
print(f"\nWeights CSV saved → {csv_path}")

# ---------------------------------------------------------------------------
# Combined 2×2 figure for paper
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "Bot":  "#E74C3C",
    "Real": "#2ECC71",
    "Scam": "#E67E22",
    "Spam": "#9B59B6",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle(
    "LIME Explanations – BERT Account Classifier (LIMFAAD)\n"
    "Positive weights (green) support the predicted class; "
    "negative weights (red) oppose it.",
    fontsize=13, fontweight='bold', y=1.01
)

for ax, (cls_idx, info) in zip(axes.flatten(), selected.items()):
    cls_name  = CLASS_NAMES[cls_idx]
    pred_idx  = CLASS_NAMES.index(info["pred_label"])
    exp       = explanations[cls_idx]

    raw = exp.as_list(label=pred_idx)
    raw_sorted = sorted(raw, key=lambda x: abs(x[1]), reverse=True)[:TOP_N_WORDS]
    raw_sorted = sorted(raw_sorted, key=lambda x: x[1])   # ascending for horizontal bar

    words   = [r[0] for r in raw_sorted]
    weights = [r[1] for r in raw_sorted]
    colors  = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

    bars = ax.barh(words, weights, color=colors, edgecolor="white", linewidth=0.5)

    # vertical zero line
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # annotate bar values
    for bar, w in zip(bars, weights):
        x_pos = w + (0.002 if w >= 0 else -0.002)
        ha    = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{w:+.3f}", va="center", ha=ha, fontsize=8)

    match_str = "✓ Correct" if info["true_label"] == info["pred_label"] else "✗ Mismatch"
    title_color = CLASS_COLORS.get(cls_name, "black")
    ax.set_title(
        f"Predicted: {info['pred_label']}  |  True: {info['true_label']}  "
        f"|  {match_str}\nConfidence: {info['confidence']:.1%}",
        fontsize=10, color=title_color, fontweight="bold", pad=6
    )

    ax.set_xlabel("LIME Feature Weight", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Class badge
    ax.text(
        0.98, 0.02, cls_name,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="white",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=title_color, alpha=0.85),
    )

# Legend
pos_patch = mpatches.Patch(color="#2ecc71", label="Supports prediction")
neg_patch = mpatches.Patch(color="#e74c3c", label="Opposes prediction")
fig.legend(
    handles=[pos_patch, neg_patch],
    loc="lower center", ncol=2,
    fontsize=10, frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout()
out_png = os.path.join(LIME_OUTPUT_DIR, "lime_bert_word_importance.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Main figure saved → {out_png}")

# ---------------------------------------------------------------------------
# Per-class individual figures (cleaner, bigger for supplementary material)
# ---------------------------------------------------------------------------
for cls_idx, info in selected.items():
    cls_name  = CLASS_NAMES[cls_idx]
    pred_idx  = CLASS_NAMES.index(info["pred_label"])
    exp       = explanations[cls_idx]

    raw        = exp.as_list(label=pred_idx)
    raw_sorted = sorted(raw, key=lambda x: abs(x[1]), reverse=True)[:TOP_N_WORDS]
    raw_sorted = sorted(raw_sorted, key=lambda x: x[1])

    words   = [r[0] for r in raw_sorted]
    weights = [r[1] for r in raw_sorted]
    colors  = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

    fig_s, ax_s = plt.subplots(figsize=(9, 6))
    bars = ax_s.barh(words, weights, color=colors, edgecolor="white", linewidth=0.5)
    ax_s.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    for bar, w in zip(bars, weights):
        x_pos = w + (0.003 if w >= 0 else -0.003)
        ha    = "left" if w >= 0 else "right"
        ax_s.text(x_pos, bar.get_y() + bar.get_height() / 2,
                  f"{w:+.4f}", va="center", ha=ha, fontsize=9)

    match_str = "Correct" if info["true_label"] == info["pred_label"] else "Mismatch"
    ax_s.set_title(
        f"LIME – BERT Account Classifier\n"
        f"Predicted: {info['pred_label']}  |  True: {info['true_label']}  "
        f"|  {match_str}  |  Confidence: {info['confidence']:.1%}",
        fontsize=11, fontweight="bold"
    )
    ax_s.set_xlabel("LIME Feature Weight", fontsize=10)
    ax_s.tick_params(axis="y", labelsize=10)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)

    pos_p = mpatches.Patch(color="#2ecc71", label="Supports prediction")
    neg_p = mpatches.Patch(color="#e74c3c", label="Opposes prediction")
    ax_s.legend(handles=[pos_p, neg_p], fontsize=9, loc="lower right")

    indiv_path = os.path.join(LIME_OUTPUT_DIR, f"lime_bert_{cls_name.lower()}_individual.png")
    plt.tight_layout()
    plt.savefig(indiv_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Individual figure saved → {indiv_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("LIME analysis complete. Output files:")
for f in sorted(os.listdir(LIME_OUTPUT_DIR)):
    print(f"  task3_limfaad/outputs/lime/{f}")
print("="*60)
