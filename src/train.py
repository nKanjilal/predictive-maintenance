"""Steps 2-5: handle imbalance, train/compare models, find drivers, pick an alert threshold.

Run from the repo root:  python src/train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import ROOT, ALL_FEATURES, RAW_FEATURES, load_dataset

SEED = 42
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150,
})

# --- Maintenance cost assumptions (edit to match a real plant) -------------------
COST_MISSED_FAILURE = 10_000   # unplanned breakdown: downtime, scrap, emergency repair
COST_ALERT = 500               # a planned inspection / tool change triggered by an alert
# -------------------------------------------------------------------------------


def build_models(pos_weight):
    """Imbalance is handled with class weights (no synthetic rows needed)."""
    return {
        "Logistic Regression (baseline)": (RAW_FEATURES, make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))),
        "Logistic Regression + engineered features": (ALL_FEATURES, make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))),
        "Random Forest": (ALL_FEATURES, RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)),
        "XGBoost": (ALL_FEATURES, XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="aucpr",
            random_state=SEED, n_jobs=-1)),
    }


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "threshold": round(float(thr), 3),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
    }


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])])


def cost_curve(y, p, thresholds):
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        cost = fn * COST_MISSED_FAILURE + (tp + fp) * COST_ALERT
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "alerts": tp + fp,
                     "recall": tp / max(tp + fn, 1), "cost": cost})
    return pd.DataFrame(rows)


def main():
    df, y, modes = load_dataset()
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    print(f"Train {len(idx_tr)} rows ({y_tr.sum()} failures) | Test {len(idx_te)} rows "
          f"({y_te.sum()} failures) | negative:positive = {pos_weight:.1f}:1")

    # Naive model for context: predict "no failure" always
    print(f"Always-'no failure' accuracy on test: {1 - y_te.mean():.1%} with 0% recall\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, test_probs, fitted = {}, {}, {}
    for name, (feats, model) in build_models(pos_weight).items():
        X_tr, X_te = df.loc[idx_tr, feats], df.loc[idx_te, feats]
        # Choose the decision threshold on out-of-fold TRAIN predictions (never on test)
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_tr, oof)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        results[name] = {"default_0.5": metrics_at(y_te, p, 0.5), "tuned": metrics_at(y_te, p, thr)}
        test_probs[name], fitted[name] = p, (feats, model, oof)
        m = results[name]["tuned"]
        print(f"{name:45s} thr={thr:.2f}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} Acc={m['accuracy']:.3f}")

    table = pd.DataFrame({k: v["tuned"] for k, v in results.items()}).T
    table.to_csv(ROOT / "reports" / "model_comparison.csv", float_format="%.4f")

    best = "XGBoost" if table.loc["XGBoost", "pr_auc"] >= table.loc["Random Forest", "pr_auc"] else "Random Forest"
    feats, model, oof = fitted[best]
    p_best = test_probs[best]
    print(f"\nBest model by PR-AUC: {best}")

    # ---- Figure: precision-recall curves ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, c in [("Logistic Regression (baseline)", MUTED), ("Random Forest", ORANGE), ("XGBoost", BLUE)]:
        pr, rc, _ = precision_recall_curve(y_te, test_probs[name])
        ax.plot(rc, pr, color=c, lw=2, label=f"{name} (PR-AUC {results[name]['tuned']['pr_auc']:.2f})")
    ax.axhline(y_te.mean(), color=MUTED, lw=1, ls=":", label=f"Random guess ({y_te.mean():.1%})")
    ax.set_xlabel("Recall (share of failures caught)"); ax.set_ylabel("Precision (share of alerts that are real)")
    ax.set_title("Tree models dominate the baseline at every recall level", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "04_precision_recall.png", bbox_inches="tight")

    # ---- Figure: metric comparison ----
    show = ["Logistic Regression (baseline)", "Random Forest", "XGBoost"]
    mets = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.26
    for i, (name, c) in enumerate(zip(show, [MUTED, ORANGE, BLUE])):
        vals = [table.loc[name, m] for m in mets]
        bars = ax.bar(np.arange(3) + (i - 1) * w, vals, w - 0.02, color=c, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), ["Precision", "Recall", "F1"]); ax.set_ylim(0, 1.1)
    ax.set_title("Test-set performance on failures (tuned thresholds)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "05_model_comparison.png", bbox_inches="tight")

    # ---- Step 4: drivers (permutation importance on the test set) ----
    X_te = df.loc[idx_te, feats]
    perm = permutation_importance(model, X_te, y_te, scoring="average_precision",
                                  n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=feats).sort_values()
    imp.to_csv(ROOT / "reports" / "feature_importance.csv", header=["pr_auc_drop"])
    labels = {"air_temp_k": "Air temperature", "process_temp_k": "Process temperature",
              "rpm": "Rotational speed", "torque_nm": "Torque", "tool_wear_min": "Tool wear",
              "type_L": "Type L", "type_M": "Type M", "type_H": "Type H",
              "temp_diff_k": "Temp difference (process − air)", "power_w": "Power (torque × speed)",
              "strain_nm_min": "Strain (torque × tool wear)"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([labels[f] for f in imp.index], imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title(f"What drives failure predictions ({best})", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "06_feature_importance.png", bbox_inches="tight")
    print("\nPermutation importance (PR-AUC drop):\n", imp.sort_values(ascending=False).round(4).to_string())

    # ---- Step 5: operations recommendation — pick alert threshold by expected cost ----
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    oof_cost = cost_curve(y_tr.values, oof, grid)
    alert_thr = float(oof_cost.loc[oof_cost.cost.idxmin(), "threshold"])
    test_cost = cost_curve(y_te.values, p_best, grid)
    chosen = test_cost.set_index("threshold").loc[alert_thr]
    no_model_cost = int(y_te.sum()) * COST_MISSED_FAILURE

    # Which failure modes are caught?
    pred_te = p_best >= alert_thr
    modes_te = modes.loc[idx_te]
    # Only count rows labeled as failures (the raw data has RNF flags on some non-failure rows)
    is_fail = y_te.values == 1
    by_mode = {m: {"failures": int((is_fail & (modes_te[m].values == 1)).sum()),
                   "caught": int((pred_te & is_fail & (modes_te[m].values == 1)).sum())}
               for m in modes_te}
    no_mode = is_fail & (modes_te.sum(axis=1).values == 0)
    by_mode["unlabeled"] = {"failures": int(no_mode.sum()), "caught": int((pred_te & no_mode).sum())}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(test_cost.threshold, test_cost.cost / 1000, color=BLUE, lw=2, label="Model-driven alerts")
    ax.axhline(no_model_cost / 1000, color=ORANGE, lw=2, ls="--", label="Run-to-failure (no model)")
    ax.axvline(alert_thr, color=MUTED, lw=1, ls=":")
    ax.annotate(f"Chosen threshold {alert_thr:.2f}", (alert_thr, chosen.cost / 1000),
                xytext=(alert_thr + 0.08, chosen.cost / 1000 + 60), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("Alert threshold (predicted failure probability)")
    ax.set_ylabel("Maintenance + failure cost ($K, test set)")
    ax.set_title("Expected cost by alert threshold", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(FIG / "07_threshold_cost.png", bbox_inches="tight")

    cm = confusion_matrix(y_te, pred_te)
    ops = {
        "model": best, "alert_threshold": alert_thr,
        "assumptions": {"cost_missed_failure": COST_MISSED_FAILURE, "cost_per_alert": COST_ALERT},
        "test_failures": int(y_te.sum()), "caught": int(chosen.tp), "missed": int(chosen.fn),
        "false_alarms": int(chosen.fp), "alerts": int(chosen.alerts),
        "recall": float(chosen.recall), "precision": float(chosen.tp / max(chosen.alerts, 1)),
        "alerts_per_1000_cycles": float(chosen.alerts / len(y_te) * 1000),
        "cost_with_model": int(chosen.cost), "cost_run_to_failure": no_model_cost,
        "savings_pct": float(1 - chosen.cost / no_model_cost),
        "confusion_matrix": cm.tolist(), "caught_by_failure_mode": by_mode,
    }
    print("\nOperations recommendation:\n", json.dumps(ops, indent=2))

    # Threshold trade-off table for the README
    tradeoff = test_cost[test_cost.threshold.isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])].copy()
    tradeoff["precision"] = tradeoff.tp / tradeoff.alerts.clip(lower=1)
    tradeoff.to_csv(ROOT / "reports" / "threshold_tradeoff.csv", index=False, float_format="%.3f")

    summary = {
        "baseline": "Logistic Regression (baseline)", "best_model": best,
        "f1_gain_points": 100 * (table.loc[best, "f1"] - table.loc["Logistic Regression (baseline)", "f1"]),
        "comparison": table.round(4).to_dict(orient="index"),
        "default_threshold_results": {k: v["default_0.5"] for k, v in results.items()},
        "operations": ops,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    # Refit the best model on ALL data for the app, keep the chosen threshold
    final = build_models(pos_weight)[best][1].fit(df[feats], y)
    joblib.dump({"model": final, "features": feats, "threshold": alert_thr,
                 "f1_threshold": results[best]["tuned"]["threshold"]}, MODELS / "model.joblib")
    print("\nSaved model to", MODELS / "model.joblib")


if __name__ == "__main__":
    main()
