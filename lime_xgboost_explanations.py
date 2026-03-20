"""
LIME (Local Interpretable Model-Agnostic Explanations) for the XGBoost account classifier.

Loads the trained XGBoost model, selects one correctly-classified example per class
(Bot, Real, Scam, Spam), applies LimeTabularExplainer, and produces:

  task3_limfaad/outputs/lime/lime_bert_word_importance.png   ← main 2×2 figure (replaces old)
  task3_limfaad/outputs/lime/lime_xgb_<class>_individual.png ← per-class figures
  task3_limfaad/outputs/lime/lime_xgb_weights_summary.csv    ← all feature weights

Run:
    python lime_xgboost_explanations.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config – must match train_limfaad_model.py exactly
# ---------------------------------------------------------------------------
XGB_MODEL_PATH   = "task3_limfaad/models/limfaad_xgboost_model.json"
LABEL_ENC_PATH   = "task3_limfaad/models/limfaad_label_encoder.pkl"
FEATURE_INFO_PATH= "task3_limfaad/models/limfaad_feature_info.pkl"
LIME_OUTPUT_DIR  = "task3_limfaad/outputs/lime"
INPUT_FILE       = "LIMFADD.csv"
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
N_LIME_SAMPLES   = 500
TOP_N_FEATURES   = 10

FEATURE_COLUMNS = [
    'Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers',
    'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads'
]

CLASS_NAMES = ['Bot', 'Real', 'Scam', 'Spam']

CLASS_COLORS = {
    "Bot":  "#E74C3C",
    "Real": "#2ECC71",
    "Scam": "#E67E22",
    "Spam": "#9B59B6",
}

os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load XGBoost model
# ---------------------------------------------------------------------------
print("Loading XGBoost model …")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

with open(LABEL_ENC_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

print(f"Classes: {label_encoder.classes_}")
print("Model loaded.")


# ---------------------------------------------------------------------------
# Preprocessing (same as train_limfaad_model.py)
# ---------------------------------------------------------------------------
def load_and_preprocess_limfaad(csv_path: str):
    df = pd.read_csv(csv_path)
    dp = df.copy()

    dp['Following/Followers'] = pd.to_numeric(dp['Following/Followers'], errors='coerce')
    dp['Posts/Followers']     = pd.to_numeric(dp['Posts/Followers'],     errors='coerce')

    dp.loc[(dp['Following/Followers'].isna()) & (dp['Followers'] == 0),
           'Following/Followers'] = dp['Following'].max()
    dp.loc[(dp['Posts/Followers'].isna()) & (dp['Followers'] == 0),
           'Posts/Followers'] = 0

    dp['Following/Followers'].fillna(dp['Following/Followers'].median(), inplace=True)
    dp['Posts/Followers'].fillna(dp['Posts/Followers'].median(), inplace=True)

    for col in ['Bio', 'Profile Picture', 'External Link', 'Threads']:
        if col in dp.columns:
            dp[col] = dp[col].apply(
                lambda v: 1 if str(v).strip().lower() in ('yes', 'y', '1', '1.0') else 0
            )

    dp['Mutual Friends'] = pd.to_numeric(dp['Mutual Friends'], errors='coerce').fillna(0)

    label_col = 'Labels' if 'Labels' in dp.columns else dp.columns[-1]
    dp = dp.dropna(subset=[label_col])

    le = LabelEncoder()
    dp['label'] = le.fit_transform(dp[label_col].str.strip())

    X = dp[FEATURE_COLUMNS].copy().values.astype(float)
    y = dp['label'].values
    return X, y, le


# ---------------------------------------------------------------------------
# XGBoost predict function for LIME
# ---------------------------------------------------------------------------
def xgb_predict_proba(X_raw: np.ndarray) -> np.ndarray:
    """Returns (N, 4) softmax probability array."""
    return xgb_model.predict_proba(X_raw)


# ---------------------------------------------------------------------------
# Load and split data
# ---------------------------------------------------------------------------
print("Loading and preprocessing LIMFAAD …")
X, y, le = load_and_preprocess_limfaad(INPUT_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# Select one correctly-classified example per class (highest confidence)
# ---------------------------------------------------------------------------
print("Selecting representative examples …")
probs_all = xgb_predict_proba(X_test)
y_pred    = np.argmax(probs_all, axis=1)
conf_all  = np.max(probs_all, axis=1)

selected = {}
for cls_idx in range(len(CLASS_NAMES)):
    cls_name = CLASS_NAMES[cls_idx]
    mask = (y_test == cls_idx) & (y_pred == cls_idx)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        idxs = np.where(y_test == cls_idx)[0]
    best = idxs[np.argmax(conf_all[idxs])]
    selected[cls_idx] = {
        "X_row":      X_test[best],
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
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=FEATURE_COLUMNS,
    class_names=CLASS_NAMES,
    mode='classification',
    discretize_continuous=False,
    random_state=RANDOM_STATE,
)

explanations = {}
all_weights  = []

for cls_idx, info in selected.items():
    cls_name = CLASS_NAMES[cls_idx]
    pred_idx = CLASS_NAMES.index(info["pred_label"])
    print(f"\nRunning LIME for class '{cls_name}' ({N_LIME_SAMPLES} samples) …")

    exp = explainer.explain_instance(
        info["X_row"],
        xgb_predict_proba,
        labels=[pred_idx],
        num_features=TOP_N_FEATURES,
        num_samples=N_LIME_SAMPLES,
    )
    explanations[cls_idx] = exp

    for feat, weight in exp.as_list(label=pred_idx):
        all_weights.append({
            "class":      cls_name,
            "feature":    feat,
            "weight":     weight,
            "true_label": info["true_label"],
            "pred_label": info["pred_label"],
            "confidence": info["confidence"],
        })

# Save CSV summary
weights_df = pd.DataFrame(all_weights)
csv_path   = os.path.join(LIME_OUTPUT_DIR, "lime_xgb_weights_summary.csv")
weights_df.to_csv(csv_path, index=False)
print(f"\nWeights CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# Combined 2×2 figure (replaces lime_bert_word_importance.png)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle(
    "LIME Explanations – XGBoost Account Classifier (LIMFAAD)\n"
    "Positive weights (green) support the predicted class; "
    "negative weights (red) oppose it.",
    fontsize=13, fontweight='bold', y=1.01
)

for ax, (cls_idx, info) in zip(axes.flatten(), selected.items()):
    cls_name = CLASS_NAMES[cls_idx]
    pred_idx = CLASS_NAMES.index(info["pred_label"])
    exp      = explanations[cls_idx]

    raw = exp.as_list(label=pred_idx)
    raw_sorted = sorted(raw, key=lambda x: abs(x[1]), reverse=True)[:TOP_N_FEATURES]
    raw_sorted = sorted(raw_sorted, key=lambda x: x[1])   # ascending for horizontal bar

    feats   = [r[0] for r in raw_sorted]
    weights = [r[1] for r in raw_sorted]
    colors  = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

    bars = ax.barh(feats, weights, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    for bar, w in zip(bars, weights):
        x_pos = w + (0.002 if w >= 0 else -0.002)
        ha    = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{w:+.3f}", va="center", ha=ha, fontsize=8)

    match_str   = "✓ Correct" if info["true_label"] == info["pred_label"] else "✗ Mismatch"
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
# Per-class individual figures
# ---------------------------------------------------------------------------
for cls_idx, info in selected.items():
    cls_name = CLASS_NAMES[cls_idx]
    pred_idx = CLASS_NAMES.index(info["pred_label"])
    exp      = explanations[cls_idx]

    raw        = exp.as_list(label=pred_idx)
    raw_sorted = sorted(raw, key=lambda x: abs(x[1]), reverse=True)[:TOP_N_FEATURES]
    raw_sorted = sorted(raw_sorted, key=lambda x: x[1])

    feats   = [r[0] for r in raw_sorted]
    weights = [r[1] for r in raw_sorted]
    colors  = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

    fig_s, ax_s = plt.subplots(figsize=(9, 6))
    bars = ax_s.barh(feats, weights, color=colors, edgecolor="white", linewidth=0.5)
    ax_s.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    for bar, w in zip(bars, weights):
        x_pos = w + (0.003 if w >= 0 else -0.003)
        ha    = "left" if w >= 0 else "right"
        ax_s.text(x_pos, bar.get_y() + bar.get_height() / 2,
                  f"{w:+.4f}", va="center", ha=ha, fontsize=9)

    match_str = "Correct" if info["true_label"] == info["pred_label"] else "Mismatch"
    ax_s.set_title(
        f"LIME – XGBoost Account Classifier\n"
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

    indiv_path = os.path.join(LIME_OUTPUT_DIR, f"lime_xgb_{cls_name.lower()}_individual.png")
    plt.tight_layout()
    plt.savefig(indiv_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Individual figure saved → {indiv_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LIME analysis complete. Output files:")
for fn in sorted(os.listdir(LIME_OUTPUT_DIR)):
    print(f"  task3_limfaad/outputs/lime/{fn}")
print("=" * 60)
