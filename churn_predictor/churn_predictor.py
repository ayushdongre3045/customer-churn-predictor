"""
churn_predictor.py
------------------
Customer Churn Prediction Pipeline
Binary Classification using Logistic Regression & Random Forest

Author : Customer Analytics Team
Dataset: data/customers.csv (run generate_data.py first)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ─────────────────────────────────────────────
# 0. Aesthetic Config
# ─────────────────────────────────────────────
PALETTE = {
    "bg":        "#0D0F14",
    "panel":     "#141720",
    "border":    "#1E2330",
    "accent1":   "#00FFB2",   # teal / churn-0
    "accent2":   "#FF4D6D",   # red  / churn-1
    "accent3":   "#7B61FF",   # violet
    "text":      "#E8EAF0",
    "muted":     "#5A6080",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "font.size":         10,
})

Path("outputs").mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load & Inspect
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("  CUSTOMER CHURN PREDICTOR  |  Binary Classification")
print("═" * 60)

df = pd.read_csv("data/customers.csv")
print(f"\n[DATA]  Shape: {df.shape}  |  Churn rate: {df['churn'].mean():.1%}")
print(f"        Missing values: {df.isnull().sum().sum()} cells\n")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
df = df.drop(columns=["customer_id"])

CATEGORICAL = ["gender", "has_partner", "has_dependents",
               "contract", "payment_method", "internet_service"]
NUMERICAL   = ["age", "tenure_months", "monthly_charges",
               "total_charges", "support_calls", "num_products",
               "senior_citizen"]

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}")

# Column transformer
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,   NUMERICAL),
    ("cat", categorical_transformer, CATEGORICAL),
])

# ─────────────────────────────────────────────
# 3. Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=500, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42
    ),
}

pipelines = {
    name: Pipeline([("prep", preprocessor), ("clf", clf)])
    for name, clf in models.items()
}

# ─────────────────────────────────────────────
# 4. Train & Evaluate
# ─────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[CROSS-VALIDATION] 5-fold ROC-AUC\n")
for name, pipe in pipelines.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  {name:<25}  AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "pipe":    pipe,
        "y_pred":  y_pred,
        "y_proba": y_proba,
        "auc":     roc_auc_score(y_test, y_proba),
        "ap":      average_precision_score(y_test, y_proba),
        "report":  classification_report(y_test, y_pred, output_dict=True),
        "cm":      confusion_matrix(y_test, y_pred),
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
    }

# ─────────────────────────────────────────────
# 5. Print Reports
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  TEST SET RESULTS")
print("─" * 60)
for name, r in results.items():
    rep = r["report"]
    print(f"\n▸ {name}")
    print(f"  ROC-AUC  : {r['auc']:.4f}   |  Avg Precision: {r['ap']:.4f}")
    print(f"  Precision: {rep['1']['precision']:.4f}  |  Recall: {rep['1']['recall']:.4f}  |  F1: {rep['1']['f1-score']:.4f}")
    print(classification_report(y_test, r["y_pred"]))

# Best model
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\n★  Best Model: {best_name}  (AUC = {best['auc']:.4f})\n")

# ─────────────────────────────────────────────
# 6. Feature Importance (Random Forest)
# ─────────────────────────────────────────────
rf_pipe  = pipelines["Random Forest"]
ohe_cols = (rf_pipe.named_steps["prep"]
            .named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(CATEGORICAL))
feat_names = np.array(NUMERICAL + list(ohe_cols))
importances = rf_pipe.named_steps["clf"].feature_importances_
fi_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
           .sort_values("importance", ascending=False)
           .head(15))

# ─────────────────────────────────────────────
# 7. Dashboard Figure
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22), facecolor=PALETTE["bg"])
fig.suptitle("CUSTOMER CHURN PREDICTOR", fontsize=22, fontweight="bold",
             color=PALETTE["accent1"], y=0.98, fontfamily="monospace")
fig.text(0.5, 0.965, "Binary Classification  ·  Logistic Regression  ·  Random Forest  ·  Gradient Boosting",
         ha="center", fontsize=10, color=PALETTE["muted"])

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38,
                       top=0.94, bottom=0.04, left=0.06, right=0.97)

# ── 7a. ROC Curves ──────────────────────────
ax_roc = fig.add_subplot(gs[0, :2])
colors  = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"]]
for (name, r), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax_roc.plot(fpr, tpr, color=col, lw=2,
                label=f"{name}  (AUC={r['auc']:.3f})")
ax_roc.plot([0,1],[0,1], "--", color=PALETTE["muted"], lw=1)
ax_roc.fill_between(*roc_curve(y_test, best["y_proba"])[:2],
                    alpha=0.06, color=PALETTE["accent1"])
ax_roc.set(title="ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax_roc.legend(loc="lower right", fontsize=9)
ax_roc.grid(True, alpha=0.3)

# ── 7b. CV AUC Summary ──────────────────────
ax_cv = fig.add_subplot(gs[0, 2])
names_short = ["LR", "RF", "GB"]
cv_means = [r["cv_mean"] for r in results.values()]
cv_stds  = [r["cv_std"]  for r in results.values()]
bars = ax_cv.bar(names_short, cv_means, color=colors, alpha=0.85,
                 yerr=cv_stds, capsize=5, error_kw={"color": PALETTE["muted"]})
for bar, val in zip(bars, cv_means):
    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f"{val:.3f}", ha="center", va="bottom", fontsize=9,
               color=PALETTE["text"])
ax_cv.set(title="5-Fold CV AUC", ylim=(0.6, 1.0))
ax_cv.grid(True, axis="y", alpha=0.3)

# ── 7c–7e. Confusion Matrices ───────────────
for idx, ((name, r), col) in enumerate(zip(results.items(), colors)):
    ax_cm = fig.add_subplot(gs[1, idx])
    cm_norm = r["cm"].astype("float") / r["cm"].sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=r["cm"], fmt="d", ax=ax_cm,
                cmap=sns.light_palette(col, as_cmap=True),
                linewidths=1, linecolor=PALETTE["border"],
                cbar=False, annot_kws={"size": 13, "weight": "bold"})
    ax_cm.set(title=name.replace(" ", "\n"), xlabel="Predicted", ylabel="Actual")
    ax_cm.set_xticklabels(["No Churn", "Churn"], rotation=0)
    ax_cm.set_yticklabels(["No Churn", "Churn"], rotation=0)

# ── 7f. Precision-Recall Curves ─────────────
ax_pr = fig.add_subplot(gs[2, :2])
for (name, r), col in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
    ax_pr.plot(rec, prec, color=col, lw=2,
               label=f"{name}  (AP={r['ap']:.3f})")
ax_pr.axhline(y_test.mean(), linestyle="--", color=PALETTE["muted"], lw=1,
              label="Baseline (random)")
ax_pr.set(title="Precision–Recall Curves", xlabel="Recall", ylabel="Precision")
ax_pr.legend(fontsize=9)
ax_pr.grid(True, alpha=0.3)

# ── 7g. Metrics Table ───────────────────────
ax_tbl = fig.add_subplot(gs[2, 2])
ax_tbl.axis("off")
model_labels = ["LR", "RF", "GB"]
metric_data  = []
for name, r in results.items():
    rep = r["report"]
    metric_data.append([
        f"{rep['1']['precision']:.3f}",
        f"{rep['1']['recall']:.3f}",
        f"{rep['1']['f1-score']:.3f}",
        f"{r['auc']:.3f}",
    ])
tbl = ax_tbl.table(
    cellText=metric_data,
    rowLabels=model_labels,
    colLabels=["Prec", "Recall", "F1", "AUC"],
    cellLoc="center", rowLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor(PALETTE["panel"] if row > 0 else PALETTE["border"])
    cell.set_edgecolor(PALETTE["border"])
    cell.set_text_props(color=PALETTE["text"] if row > 0 else PALETTE["accent1"])
ax_tbl.set_title("Test Metrics by Model", color=PALETTE["text"], pad=12)

# ── 7h. Feature Importance ──────────────────
ax_fi = fig.add_subplot(gs[3, :])
bar_colors = [PALETTE["accent1"] if i < 5 else PALETTE["accent3"] if i < 10
              else PALETTE["muted"] for i in range(len(fi_df))]
bars2 = ax_fi.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
                   color=bar_colors[::-1], alpha=0.88)
for bar in bars2:
    w = bar.get_width()
    ax_fi.text(w + 0.001, bar.get_y() + bar.get_height()/2,
               f"{w:.4f}", va="center", fontsize=8, color=PALETTE["muted"])
ax_fi.set(title="Top 15 Feature Importances  (Random Forest)",
          xlabel="Importance Score")
ax_fi.grid(True, axis="x", alpha=0.3)

plt.savefig("outputs/churn_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"])
print("Dashboard saved → outputs/churn_dashboard.png")
plt.close()

# ─────────────────────────────────────────────
# 8. Predict New Customer (Demo)
# ─────────────────────────────────────────────
new_customer = pd.DataFrame([{
    "gender": "Male", "senior_citizen": 0, "has_partner": "No",
    "has_dependents": "No", "age": 34, "tenure_months": 3,
    "contract": "Month-to-month", "payment_method": "Electronic check",
    "internet_service": "Fiber optic", "num_products": 1,
    "monthly_charges": 95.50, "total_charges": 285.00, "support_calls": 4,
}])

best_pipe = pipelines[best_name]
churn_prob_new = best_pipe.predict_proba(new_customer)[0, 1]
churn_pred_new = best_pipe.predict(new_customer)[0]

print("─" * 60)
print("  DEMO PREDICTION  (new customer)")
print("─" * 60)
print(f"  Model         : {best_name}")
print(f"  Churn Prob    : {churn_prob_new:.1%}")
print(f"  Prediction    : {'⚠  WILL CHURN' if churn_pred_new else '✓  WILL STAY'}")
print("─" * 60 + "\n")
