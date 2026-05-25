# generates all thesis figures from pipeline.py csv outputs
# run pipeline.py first, then this script

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = "thesis_figures_clean"
os.makedirs(OUT_DIR, exist_ok=True)

# Okabe-Ito palette (colorblind safe)
CB_ORANGE    = "#E69F00"
CB_SKY       = "#56B4E9"
CB_GREEN     = "#009E73"
CB_YELLOW    = "#F0E442"
CB_BLUE      = "#0072B2"
CB_VERMILLION = "#D55E00"
CB_PURPLE    = "#CC79A7"
CB_BLACK     = "#000000"
CB_GRAY      = "#999999"


def _save(fig, name, dpi=220):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig2_holdout():
    results = pd.read_csv("holdout_results.csv")
    label_map = {
        "Naive":        "Naive (global mean)",
        "Roll-5":       "Rolling 5-match mean",
        "Poisson GLM":  "Poisson GLM",
        "XGBoost S3":   "XGBoost S3",
        "XGBoost Tuned":"XGBoost Tuned",
        "Neural Net":   "Neural Net (PyTorch)",
    }
    color_map = {
        "Naive":        CB_GRAY,
        "Roll-5":       CB_SKY,
        "Poisson GLM":  CB_BLUE,
        "Neural Net":   CB_PURPLE,
        "XGBoost S3":   CB_ORANGE,
        "XGBoost Tuned":CB_GREEN,
    }
    results = results.sort_values("MAE", ascending=False).reset_index(drop=True)
    labels  = [label_map[m] for m in results["model"]]
    colors  = [color_map[m] for m in results["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(results)), results["MAE"],
                   color=colors, edgecolor="white")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Holdout MAE (shots per match)", fontsize=11)
    ax.set_title("Holdout MAE by model on the player-based test set",
                 fontsize=12, pad=12)
    for bar, val in zip(bars, results["MAE"]):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    _despine(ax)
    ax.set_xlim(0, results["MAE"].max() * 1.18)
    _save(fig, "fig2_holdout_comparison.png", dpi=200)


def fig3_feature_importance():
    imp = (pd.read_csv("feature_importance.csv")
           .sort_values("gain", ascending=False)
           .head(20))
    fig, ax = plt.subplots(figsize=(11, 8.5))
    y_pos = range(len(imp))[::-1]
    bars  = ax.barh(y_pos, imp["gain"],
                    color=CB_ORANGE, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp["feature"], fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("Importance score (gain)", fontsize=14)
    ax.set_title("Top 20 feature importances – XGBoost Tuned",
                 fontsize=15, pad=14)
    for bar, val in zip(bars, imp["gain"]):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=12)
    _despine(ax)
    ax.set_xlim(0, imp["gain"].max() * 1.15)
    _save(fig, "fig3_feature_importance.png")


def fig4_calibration():
    pred  = pd.read_csv("predictions.csv")
    y     = pred["actual"].values
    n_bins = 10
    edges  = np.linspace(0, 1, n_bins + 1)

    fig, axes = plt.subplots(3, 1, figsize=(7, 12))

    for ax, t in zip(axes, [1, 2, 3]):
        y_bin = (y >= t).astype(int)
        raw   = pred[f"prob_xgb_raw_ge{t}"].values
        cal   = pred[f"prob_xgb_cal_ge{t}"].values

        for probs, color, marker, label in [
            (raw, CB_VERMILLION, "o", "Uncalibrated (XGBoost Tuned)"),
            (cal, CB_GREEN,      "s", "Calibrated (+ Isotonic)"),
        ]:
            xs, ys, sizes = [], [], []
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                m = (probs >= lo) & (probs < hi)
                if m.sum() == 0:
                    continue
                xs.append(probs[m].mean())
                ys.append(y_bin[m].mean())
                sizes.append(m.sum())
            sizes = np.array(sizes, dtype=float)
            sizes = 30 + 250 * (sizes / sizes.max())
            ax.plot(xs, ys, color=color, lw=1.2, alpha=0.9, zorder=2)
            ax.scatter(xs, ys, s=sizes, color=color, marker=marker,
                       alpha=0.85, edgecolor="white", linewidth=0.6,
                       label=label, zorder=3)

        ax.plot([0, 1], [0, 1], "--", color=CB_GRAY, lw=1,
                label="Perfect calibration", zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"P(shots ≥ {t})", fontsize=12)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        _despine(ax)

    fig.tight_layout()
    _save(fig, "fig4_calibration.png")


def fig5_predicted_vs_actual():
    pred      = pd.read_csv("predictions.csv")
    actual    = pred["actual"].values.astype(int)
    predicted = pred["pred_xgb_tuned"].values
    residual  = actual - predicted

    palette = {
        0: CB_BLUE,
        1: CB_VERMILLION,
        2: CB_GREEN,
        3: CB_PURPLE,
        4: CB_ORANGE,
        5: CB_SKY,
        6: CB_YELLOW,
        7: CB_BLACK,
    }
    markers = {0:"o", 1:"s", 2:"^", 3:"D", 4:"v", 5:"P", 6:"X", 7:"*"}

    fig, ax = plt.subplots(figsize=(11, 4))
    for v in sorted(set(actual)):
        m = actual == v
        ax.scatter(predicted[m], residual[m],
                   color=palette.get(v, CB_GRAY),
                   marker=markers.get(v, "o"),
                   s=14, alpha=0.7, edgecolor="none",
                   label=f"Actual = {v} (n={m.sum()})")
    ax.axhline(0, color=CB_BLACK, linestyle="--", lw=0.8)
    ax.set_xlabel("Predicted shots")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals vs predicted values, coloured by actual shot count",
                 fontsize=12, pad=10)
    ax.legend(title="Actual shots", fontsize=8, title_fontsize=9,
              loc="upper right", frameon=False, ncol=2)
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig5_predicted_vs_actual.png")


def fig6_residuals():
    pred      = pd.read_csv("predictions.csv")
    res       = pd.read_csv("residuals.csv")
    actual    = pred["actual"].values.astype(int)
    predicted = pred["pred_xgb_tuned"].values
    residual  = actual - predicted
    r         = res["residual"].values
    mean_r    = float(r.mean())

    palette = {
        0: CB_BLUE, 1: CB_VERMILLION, 2: CB_GREEN, 3: CB_PURPLE,
        4: CB_ORANGE, 5: CB_SKY, 6: CB_YELLOW, 7: CB_BLACK,
    }
    markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v", 5: "P", 6: "X", 7: "*"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    # top panel: scatter of residuals vs predicted, coloured by actual shot count
    for v in sorted(set(actual)):
        m = actual == v
        ax1.scatter(predicted[m], residual[m],
                    color=palette.get(v, CB_GRAY),
                    marker=markers.get(v, "o"),
                    s=14, alpha=0.7, edgecolor="none",
                    label=f"Actual = {v} (n={m.sum()})")
    ax1.axhline(0, color=CB_BLACK, linestyle="--", lw=0.8)
    ax1.set_xlabel("Predicted shots")
    ax1.set_ylabel("Residual (actual − predicted)")
    ax1.set_title("Residuals vs predicted values, coloured by actual shot count",
                  fontsize=12, pad=10)
    ax1.legend(title="Actual shots", fontsize=8, title_fontsize=9,
               loc="upper right", frameon=False, ncol=2)
    _despine(ax1)

    # bottom panel: histogram of all residuals
    ax2.hist(r, bins=40, color=CB_BLUE, edgecolor="white", linewidth=0.4)
    ax2.axvline(0,      color=CB_BLACK,      linestyle="--", lw=1,
                label="Zero (no error)")
    ax2.axvline(mean_r, color=CB_VERMILLION, lw=1.5,
                label=f"Mean = {mean_r:.3f}")
    ax2.set_xlabel("Residual (actual − predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of residuals", fontsize=12, pad=10)
    ax2.legend(frameon=False)
    _despine(ax2)

    fig.tight_layout()
    _save(fig, "fig6_residuals.png")


def fig7_bootstrap_ci():
    bs  = pd.read_csv("bootstrap_samples.csv")
    pos = pd.read_csv("subgroup_position.csv")

    overall  = bs["overall"].values
    mean_mae = float(overall.mean())
    lo, hi   = np.percentile(overall, [2.5, 97.5])

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    ax = axes[0]
    ax.hist(overall, bins=40, color=CB_ORANGE, edgecolor="white", linewidth=0.4)
    ax.axvline(lo,       color=CB_VERMILLION, linestyle="--", lw=1.2,
               label=f"95% CI: [{lo:.4f}, {hi:.4f}]")
    ax.axvline(hi,       color=CB_VERMILLION, linestyle="--", lw=1.2)
    ax.axvline(mean_mae, color=CB_BLACK,      lw=1.4,
               label=f"Mean MAE = {mean_mae:.4f}")
    ax.set_xlabel("Bootstrap MAE")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap distribution of MAE (1,000 resamples)",
                 fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9)
    _despine(ax)

    ax   = axes[1]
    colors = {"ATT": CB_VERMILLION, "MID": CB_GREEN, "DEF": CB_BLUE}
    hatches = {"ATT": "//", "MID": "..", "DEF": ""}
    pos  = pos.set_index("position").loc[["ATT", "MID", "DEF"]].reset_index()
    xs   = range(len(pos))
    err_lo = pos["MAE"] - pos["MAE_lo"]
    err_hi = pos["MAE_hi"] - pos["MAE"]
    bars = ax.bar(xs, pos["MAE"],
                  yerr=[err_lo, err_hi],
                  color=[colors[p] for p in pos["position"]],
                  hatch=[hatches[p] for p in pos["position"]],
                  edgecolor="white", capsize=8, linewidth=0.5)
    for bar, v in zip(bars, pos["MAE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(pos["position"])
    ax.set_ylabel("MAE")
    ax.set_title("MAE with 95% bootstrap CI by position", fontsize=12, pad=10)
    ax.set_ylim(0, pos["MAE_hi"].max() * 1.2)
    _despine(ax)

    fig.tight_layout()
    _save(fig, "fig7_bootstrap_ci.png")


def fig8_error_stratification():
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))

    def _bar(ax, labels, values, title, color=CB_ORANGE):
        bars = ax.bar(range(len(labels)), values,
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("MAE")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, max(values) * 1.25)
        _despine(ax)

    pos = pd.read_csv("subgroup_position.csv").set_index("position").loc[["ATT", "MID", "DEF"]]
    _bar(axes[0, 0], pos.index.tolist(), pos["MAE"].tolist(), "Position group",
         color=[CB_VERMILLION, CB_GREEN, CB_BLUE])

    ha = pd.read_csv("subgroup_homeaway.csv").set_index("context").loc[["Away", "Home"]]
    _bar(axes[0, 1], ha.index.tolist(), ha["MAE"].tolist(), "Home / Away",
         color=[CB_SKY, CB_ORANGE])

    sq = pd.read_csv("subgroup_squad.csv").set_index("tier").loc[["Elite", "Strong", "Weaker"]]
    _bar(axes[1, 0], sq.index.tolist(), sq["MAE"].tolist(), "Squad tier",
         color=[CB_VERMILLION, CB_ORANGE, CB_SKY])

    ph = pd.read_csv("subgroup_phase.csv")
    _bar(axes[1, 1],
         [f"{p}\n({n})" for p, n in zip(ph["phase"], ph["n"])],
         ph["MAE"].tolist(), "Season phase", color=CB_BLUE)

    bk = pd.read_csv("subgroup_bucket.csv")
    _bar(axes[2, 0], bk["bucket"].tolist(), bk["MAE"].tolist(),
         "Predicted bucket", color=CB_ORANGE)

    fm = pd.read_csv("subgroup_form.csv").set_index("form").loc[["Low", "Medium", "Good", "Hot"]]
    _bar(axes[2, 1],
         ["Low\n(≤0.5)", "Medium\n(0.5-1)", "Good\n(1-1.5)", "Hot\n(>1.5)"],
         fm["MAE"].tolist(), "Form tier",
         color=[CB_SKY, CB_GREEN, CB_ORANGE, CB_VERMILLION])

    fig.suptitle("Error stratification across subgroups (XGBoost Tuned)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save(fig, "fig8_error_stratification.png")


def fig_split_check():
    from pipeline import (
        load_and_filter, add_features, player_split, DATA_PATH,
    )
    df = add_features(load_and_filter(DATA_PATH))
    train_p, test_p = player_split(df)
    df_train = df[df["player"].isin(train_p)]
    df_test  = df[df["player"].isin(test_p)]
    y_train  = df_train["sh"].values
    y_test   = df_test["sh"].values

    max_shots   = 7
    shot_labels = [str(k) for k in range(max_shots)] + [f"{max_shots}+"]
    train_pct   = [(y_train == k).mean() * 100 for k in range(max_shots)] + \
                  [(y_train >= max_shots).mean() * 100]
    test_pct    = [(y_test == k).mean() * 100 for k in range(max_shots)] + \
                  [(y_test >= max_shots).mean() * 100]

    pos_groups = ["ATT", "MID", "DEF"]
    train_pos  = [(df_train["pos_group"] == g).mean() * 100 for g in pos_groups]
    test_pos   = [(df_test["pos_group"] == g).mean() * 100 for g in pos_groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(shot_labels)) * 1.25
    w = 0.5
    ax1.bar(x - w / 2, train_pct, w, color=CB_BLUE,   label="Training",
            edgecolor="white")
    ax1.bar(x + w / 2, test_pct,  w, color=CB_ORANGE, label="Test",
            edgecolor="white", hatch="..")
    ax1.set_xticks(x); ax1.set_xticklabels(shot_labels)
    ax1.set_xlabel("Shots in match")
    ax1.set_ylabel("Share of observations (%)")
    ax1.set_title("Shot count distribution")
    ax1.legend(frameon=False)
    _despine(ax1)

    x2 = np.arange(len(pos_groups))
    ax2.bar(x2 - w / 2, train_pos, w, color=CB_BLUE,   label="Training",
            edgecolor="white")
    ax2.bar(x2 + w / 2, test_pos,  w, color=CB_ORANGE, label="Test",
            edgecolor="white", hatch="..")
    ax2.set_xticks(x2); ax2.set_xticklabels(pos_groups)
    ax2.set_xlabel("Position group")
    ax2.set_ylabel("Share of observations (%)")
    ax2.set_title("Position composition")
    ax2.legend(frameon=False)
    _despine(ax2)

    for i, (tr, te) in enumerate(zip(train_pct, test_pct)):
        ax1.text(x[i] - w / 2, tr + 0.7, f"{tr:.1f}", ha="center", fontsize=9)
        ax1.text(x[i] + w / 2, te + 0.7, f"{te:.1f}", ha="center", fontsize=9)
    for i, (tr, te) in enumerate(zip(train_pos, test_pos)):
        ax2.text(i - w / 2, tr + 0.7, f"{tr:.1f}", ha="center", fontsize=9)
        ax2.text(i + w / 2, te + 0.7, f"{te:.1f}", ha="center", fontsize=9)

    ax1.set_ylim(0, max(max(train_pct), max(test_pct)) * 1.15)
    ax2.set_ylim(0, max(max(train_pos), max(test_pos)) * 1.15)
    fig.tight_layout()
    _save(fig, "fig_split_check.png")


def main():
    needed = [
        "holdout_results.csv", "feature_importance.csv",
        "predictions.csv", "residuals.csv",
        "bootstrap_samples.csv",
        "subgroup_position.csv", "subgroup_homeaway.csv",
        "subgroup_squad.csv", "subgroup_phase.csv",
        "subgroup_bucket.csv", "subgroup_form.csv",
    ]
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        raise SystemExit(
            "Missing pipeline outputs: " + ", ".join(missing) +
            "\nRun:  python pipeline.py  first."
        )

    fig2_holdout()
    fig3_feature_importance()
    fig4_calibration()
    fig5_predicted_vs_actual()
    fig6_residuals()
    fig7_bootstrap_ci()
    fig8_error_stratification()
    fig_split_check()


if __name__ == "__main__":
    main()
