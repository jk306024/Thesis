import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

import random

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

DATA_PATH = "FINALDATASET.csv"
SEED = 42

# seed everything so results are reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TRAIN_FRAC = 0.80
N_FOLDS = 5
N_BOOTSTRAP = 1000

STAT_COLS = ["sh", "sca", "Att 3rd", "Att Pen", "PrgR", "1/3", "CPA"]
EWM_COLS = ["sh", "sca", "Att 3rd", "Att Pen"]
STD_COLS = ["sh", "sca"]

XGB_S3 = dict(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    objective="count:poisson", random_state=SEED, verbosity=0,
)

XGB_TUNED = dict(
    n_estimators=600,
    learning_rate=0.04250464900428301,
    max_depth=9,
    min_child_weight=9,
    subsample=0.867585689335889,
    colsample_bytree=0.7616356480376688,
    objective="count:poisson", random_state=SEED, verbosity=0,
)


def load_and_filter(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["start"] == True]
    df = df[df["pos"].notna()]
    df["pos"] = df["pos"].astype(str).str.split(",").str[0]
    df = df[df["pos"] != "GK"]
    df = df[df["sh"].notna()]
    df["sh"] = df["sh"].astype(float)
    return df.reset_index(drop=True)


def add_features(df):
    df = df.sort_values(["player", "date"]).reset_index(drop=True)

    for k in [3, 5, 10]:
        for col in STAT_COLS:
            df[f"{col}_roll{k}"] = (
                df.groupby("player")[col]
                .transform(lambda x: x.shift(1).rolling(k, min_periods=1).mean())
            )

    for k in [3, 5, 10]:
        for col in STD_COLS:
            df[f"{col}_std{k}"] = (
                df.groupby("player")[col]
                .transform(lambda x: x.shift(1).rolling(k, min_periods=2).std().fillna(0))
            )

    for s in [3, 5]:
        for col in EWM_COLS:
            df[f"{col}_ewm{s}"] = (
                df.groupby("player")[col]
                .transform(lambda x: x.shift(1).ewm(span=s, adjust=False).mean())
            )

    df["sh_momentum"] = df["sh_roll3"] - df["sh_roll10"]
    df["sh_momentum_ew"] = df["sh_ewm3"] - df["sh_ewm5"]
    df["sh_consistency"] = 1 / (df["sh_std5"] + 0.5)

    minutes_roll5 = df.groupby("player")["min"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["sh_per90_roll5"] = (df["sh_roll5"] / minutes_roll5 * 90).clip(0, 15)

    opp = (
        df.groupby(["date", "opponent"])["sh"].sum()
        .reset_index()
        .rename(columns={"opponent": "squad", "sh": "sh_against"})
        .sort_values(["squad", "date"])
    )
    opp["opp_sh_conceded_roll5"] = (
        opp.groupby("squad")["sh_against"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df = df.merge(
        opp[["date", "squad", "opp_sh_conceded_roll5"]].rename(columns={"squad": "opponent"}),
        on=["date", "opponent"], how="left",
    )

    pos_map = {
        "FW": "ATT", "RW": "ATT", "LW": "ATT", "AM": "ATT",
        "CM": "MID", "DM": "MID", "RM": "MID", "LM": "MID",
        "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
    }
    df["pos_group"] = df["pos"].map(pos_map).fillna("MID")
    df["pos_group_enc"] = LabelEncoder().fit_transform(df["pos_group"])
    df["pos_enc"] = LabelEncoder().fit_transform(df["pos"])

    df["elo_diff"] = df["elo_for_player"] - df["elo_opponent"]
    df["elo_ratio"] = df["elo_for_player"] / df["elo_opponent"].replace(0, 1)
    df["is_home"] = (df["home_away"] == "Home").astype(int)
    df["home_x_attacker"] = df["is_home"] * (df["pos_group"] == "ATT").astype(int)

    df["sh_season_avg"] = (
        df.groupby("player")["sh"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["match_num"] = df.groupby("player").cumcount() + 1

    return df


def feature_list():
    feats = []
    feats += [f"{c}_roll{k}" for c in STAT_COLS for k in [3, 5, 10]]
    feats += [f"{c}_std{k}" for c in STD_COLS for k in [3, 5, 10]]
    feats += [f"{c}_ewm{s}" for c in EWM_COLS for s in [3, 5]]
    feats += [
        "sh_momentum", "sh_momentum_ew", "sh_per90_roll5", "sh_consistency",
        "opp_sh_conceded_roll5", "sh_season_avg", "match_num",
        "elo_diff", "elo_ratio", "is_home", "home_x_attacker",
        "pos_enc", "pos_group_enc",
    ]
    return feats


def player_split(df, seed=SEED, train_frac=TRAIN_FRAC):
    players = df["player"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(players)
    n_train = int(len(players) * train_frac)
    train_set = set(players[:n_train])
    test_set = set(players[n_train:])
    return train_set, test_set


def mae_rmse(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))


def poisson_tail(rate, threshold):
    # P(X >= threshold) where X ~ Poisson(rate)
    rate = np.asarray(rate)
    under = np.zeros_like(rate, dtype=float)
    for k in range(threshold):
        under += np.exp(-rate) * (rate ** k) / math.factorial(k)
    return np.clip(1 - under, 1e-6, 1 - 1e-6)


def ece(y_binary, probs, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = len(y_binary)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(probs[m].mean() - y_binary[m].mean())
    return total


class PoissonNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_nn(X_train, y_train, X_test, n_epochs=150, lr=1e-3, patience=15):
    # reseed before each fit so cv folds and final model match
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    model = PoissonNet(X_train.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_loss, best_state, stale = float("inf"), None, 0
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        out = model(Xtr_t)
        loss = torch.mean(out - ytr_t * torch.log(out + 1e-8))
        loss.backward()
        opt.step()
        sched.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.clip(model(Xte_t).numpy(), 0, None)


def run_cv(df_train, X_train, y_train, feats):
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df_train["player"].values

    rows = []
    oof_y, oof_xgb, oof_nn = [], [], []

    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups), start=1):
        Xtr, Xva = X_train[tr], X_train[va]
        ytr, yva = y_train[tr], y_train[va]
        df_va = df_train.iloc[va]

        global_mean = ytr.mean()
        naive_mae, _ = mae_rmse(yva, np.full(len(yva), global_mean))
        roll5_mae, _ = mae_rmse(yva, df_va["sh_roll5"].fillna(global_mean).values)

        pois = PoissonRegressor(max_iter=500, alpha=0.1).fit(Xtr, ytr)
        pois_mae, _ = mae_rmse(yva, np.clip(pois.predict(Xva), 0, None))

        s3 = XGBRegressor(**XGB_S3).fit(Xtr, ytr, verbose=False)
        s3_mae, _ = mae_rmse(yva, np.clip(s3.predict(Xva), 0, None))

        tuned = XGBRegressor(**XGB_TUNED).fit(Xtr, ytr, verbose=False)
        tuned_pred = np.clip(tuned.predict(Xva), 0, None)
        tuned_mae, _ = mae_rmse(yva, tuned_pred)

        nn_pred = fit_nn(Xtr, ytr, Xva, n_epochs=100, patience=10)
        nn_mae, _ = mae_rmse(yva, nn_pred)

        oof_y.extend(yva.tolist())
        oof_xgb.extend(tuned_pred.tolist())
        oof_nn.extend(nn_pred.tolist())

        rows.append(dict(
            fold=fold,
            naive=naive_mae, roll5=roll5_mae, poisson=pois_mae,
            xgb_s3=s3_mae, xgb_tuned=tuned_mae, nn=nn_mae,
        ))
        print(f"  fold {fold}: tuned MAE {tuned_mae:.4f}, NN MAE {nn_mae:.4f}")

    cv = pd.DataFrame(rows)
    return cv, np.array(oof_y), np.array(oof_xgb), np.array(oof_nn)


def fit_calibrators(oof_y, oof_pred):
    cal = {}
    for t in [1, 2, 3]:
        y_bin = (oof_y >= t).astype(int)
        probs = poisson_tail(oof_pred, t)
        iso = IsotonicRegression(out_of_bounds="clip").fit(probs, y_bin)
        cal[t] = iso
    return cal


def calibrated_probs(rates, threshold, calibrator):
    raw = poisson_tail(rates, threshold)
    return np.clip(calibrator[threshold].predict(raw), 1e-6, 1 - 1e-6)


def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        samples.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    samples = np.array(samples)
    return samples.mean(), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def subgroup_table(df_test, y_true, y_pred, col):
    out = []
    for g, sub in df_test.groupby(col):
        idx = sub.index.values
        mask = np.isin(np.arange(len(df_test)), np.where(df_test.index.isin(idx))[0])
        if mask.sum() < 5:
            continue
        mae, rmse = mae_rmse(y_true[mask], y_pred[mask])
        out.append({col: g, "n": int(mask.sum()), "MAE": mae, "RMSE": rmse,
                    "avg_shots": float(y_true[mask].mean())})
    return pd.DataFrame(out)


def main():
    print("loading data")
    df = load_and_filter(DATA_PATH)
    print(f"  {len(df):,} rows, {df['player'].nunique()} players")

    print("building features")
    df = add_features(df)
    feats = feature_list()
    feats = [f for f in feats if f in df.columns and df[f].notna().sum() > 10]
    print(f"  {len(feats)} features")

    train_players, test_players = player_split(df)
    df_train = df[df["player"].isin(train_players)].reset_index(drop=True)
    df_test = df[df["player"].isin(test_players)].reset_index(drop=True)
    X_train = df_train[feats].fillna(0).values
    X_test = df_test[feats].fillna(0).values
    y_train = df_train["sh"].values
    y_test = df_test["sh"].values
    print(f"  train: {len(train_players)} players / {len(df_train):,} rows")
    print(f"  test:  {len(test_players)} players / {len(df_test):,} rows")

    print("cross-validation")
    cv, oof_y, oof_xgb, oof_nn = run_cv(df_train, X_train, y_train, feats)
    cv.to_csv("cv_results.csv", index=False)
    print(cv.describe().loc[["mean", "std"]].round(4).to_string())

    print("fitting calibrators on OOF predictions")
    cal_xgb = fit_calibrators(oof_y, oof_xgb)
    cal_nn = fit_calibrators(oof_y, oof_nn)

    print("training final models on full training set")
    global_mean = y_train.mean()

    pred_naive = np.full(len(y_test), global_mean)
    pred_roll5 = df_test["sh_roll5"].fillna(global_mean).values

    pois = PoissonRegressor(max_iter=500, alpha=0.1).fit(X_train, y_train)
    pred_pois = np.clip(pois.predict(X_test), 0, None)

    s3 = XGBRegressor(**XGB_S3).fit(X_train, y_train, verbose=False)
    pred_s3 = np.clip(s3.predict(X_test), 0, None)

    tuned = XGBRegressor(**XGB_TUNED).fit(X_train, y_train, verbose=False)
    pred_tuned = np.clip(tuned.predict(X_test), 0, None)

    pred_nn = fit_nn(X_train, y_train, X_test)

    print("\nholdout results")
    results = []
    for name, pred in [
        ("Naive", pred_naive), ("Roll-5", pred_roll5), ("Poisson GLM", pred_pois),
        ("XGBoost S3", pred_s3), ("XGBoost Tuned", pred_tuned), ("Neural Net", pred_nn),
    ]:
        mae, rmse = mae_rmse(y_test, pred)
        results.append({"model": name, "MAE": mae, "RMSE": rmse})
        print(f"  {name:<15} MAE {mae:.4f}  RMSE {rmse:.4f}")
    pd.DataFrame(results).to_csv("holdout_results.csv", index=False)

    mean_mae, lo, hi = bootstrap_ci(y_test, pred_tuned)
    print(f"\nbootstrap 95% CI for XGBoost Tuned: [{lo:.4f}, {hi:.4f}]  (mean {mean_mae:.4f})")

    print("\ncalibration (ECE before / after)")
    cal_rows = []
    for t in [1, 2, 3]:
        y_bin = (y_test >= t).astype(int)
        raw_xgb = poisson_tail(pred_tuned, t)
        cal_xgb_p = calibrated_probs(pred_tuned, t, cal_xgb)
        raw_nn = poisson_tail(pred_nn, t)
        cal_nn_p = calibrated_probs(pred_nn, t, cal_nn)
        cal_rows.append(dict(
            threshold=f">={t}",
            xgb_raw=ece(y_bin, raw_xgb), xgb_cal=ece(y_bin, cal_xgb_p),
            nn_raw=ece(y_bin, raw_nn), nn_cal=ece(y_bin, cal_nn_p),
        ))
        print(f"  >={t}  XGB {ece(y_bin, raw_xgb):.4f} -> {ece(y_bin, cal_xgb_p):.4f}   "
              f"NN {ece(y_bin, raw_nn):.4f} -> {ece(y_bin, cal_nn_p):.4f}")
    pd.DataFrame(cal_rows).to_csv("calibration_results.csv", index=False)

    print("\nfeature importance (XGBoost Tuned)")
    importance = pd.DataFrame({
        "feature": feats,
        "gain": tuned.feature_importances_,
    }).sort_values("gain", ascending=False)
    importance.to_csv("feature_importance.csv", index=False)
    print(importance.head(10).to_string(index=False))

    print("\nsubgroup analysis")
    df_test = df_test.reset_index(drop=True)

    pos_tab = []
    for g in ["ATT", "MID", "DEF"]:
        m = (df_test["pos_group"] == g).values
        if m.sum() == 0:
            continue
        mae, rmse = mae_rmse(y_test[m], pred_tuned[m])
        _, lo, hi = bootstrap_ci(y_test[m], pred_tuned[m])
        pos_tab.append({"position": g, "n": int(m.sum()), "MAE": mae,
                        "MAE_lo": lo, "MAE_hi": hi, "RMSE": rmse})
    pd.DataFrame(pos_tab).to_csv("subgroup_position.csv", index=False)
    print(pd.DataFrame(pos_tab).round(4).to_string(index=False))

    home_tab = []
    for ha in [1, 0]:
        m = (df_test["is_home"] == ha).values
        mae, rmse = mae_rmse(y_test[m], pred_tuned[m])
        home_tab.append({"context": "Home" if ha == 1 else "Away",
                         "n": int(m.sum()),
                         "avg_shots": float(y_test[m].mean()),
                         "MAE": mae, "RMSE": rmse})
    pd.DataFrame(home_tab).to_csv("subgroup_homeaway.csv", index=False)

    df_test["squad_tier"] = pd.cut(
        df_test["elo_for_player"],
        bins=[-np.inf, 1700, 1850, np.inf],
        labels=["Weaker", "Strong", "Elite"],
    )
    squad_tab = []
    for tier in ["Elite", "Strong", "Weaker"]:
        m = (df_test["squad_tier"] == tier).values
        if m.sum() == 0:
            continue
        mae, rmse = mae_rmse(y_test[m], pred_tuned[m])
        squad_tab.append({"tier": tier, "n": int(m.sum()),
                          "avg_shots": float(y_test[m].mean()),
                          "MAE": mae, "RMSE": rmse})
    pd.DataFrame(squad_tab).to_csv("subgroup_squad.csv", index=False)

    phase_bins = [(1, 10, "Early"), (11, 20, "Mid-early"),
                  (21, 30, "Mid-late"), (31, 100, "Late")]
    phase_tab = []
    for lo, hi, name in phase_bins:
        m = ((df_test["match_num"] >= lo) & (df_test["match_num"] <= hi)).values
        if m.sum() == 0:
            continue
        mae, rmse = mae_rmse(y_test[m], pred_tuned[m])
        phase_tab.append({"phase": name, "n": int(m.sum()), "MAE": mae, "RMSE": rmse})
    pd.DataFrame(phase_tab).to_csv("subgroup_phase.csv", index=False)

    df_test["pred_bucket"] = pd.cut(
        pred_tuned,
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
        labels=["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"],
    )
    bucket_tab = []
    for b in ["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"]:
        m = (df_test["pred_bucket"] == b).values
        if m.sum() == 0:
            continue
        mae, rmse = mae_rmse(y_test[m], pred_tuned[m])
        bucket_tab.append({"bucket": b, "n": int(m.sum()), "MAE": mae})
    pd.DataFrame(bucket_tab).to_csv("subgroup_bucket.csv", index=False)

    has_history = df_test["sh_roll5"].notna().values
    form_vals = df_test.loc[has_history, "sh_roll5"].values
    form_tiers = pd.cut(
        form_vals,
        bins=[-np.inf, 0.5, 1.0, 1.5, np.inf],
        labels=["Low", "Medium", "Good", "Hot"],
    )
    form_tab = []
    for tier in ["Hot", "Good", "Medium", "Low"]:
        sub_mask = (form_tiers == tier)
        if sub_mask.sum() == 0:
            continue
        idx = np.where(has_history)[0][np.asarray(sub_mask)]
        mae, rmse = mae_rmse(y_test[idx], pred_tuned[idx])
        form_tab.append({"form": tier, "n": int(len(idx)), "MAE": mae, "RMSE": rmse})
    pd.DataFrame(form_tab).to_csv("subgroup_form.csv", index=False)

    residuals = pd.DataFrame({
        "player": df_test["player"].values,
        "date": df_test["date"].values,
        "actual": y_test,
        "predicted": pred_tuned,
        "residual": y_test - pred_tuned,
        "pos_group": df_test["pos_group"].values,
    })
    residuals.to_csv("residuals.csv", index=False)

    # save all predictions + calibrated probs for figures
    predictions = pd.DataFrame({
        "player": df_test["player"].values,
        "date": df_test["date"].values,
        "pos_group": df_test["pos_group"].values,
        "actual": y_test,
        "pred_naive": pred_naive,
        "pred_roll5": pred_roll5,
        "pred_poisson": pred_pois,
        "pred_xgb_s3": pred_s3,
        "pred_xgb_tuned": pred_tuned,
        "pred_nn": pred_nn,
    })
    for t in [1, 2, 3]:
        predictions[f"prob_xgb_raw_ge{t}"] = poisson_tail(pred_tuned, t)
        predictions[f"prob_xgb_cal_ge{t}"] = calibrated_probs(pred_tuned, t, cal_xgb)
        predictions[f"prob_nn_raw_ge{t}"] = poisson_tail(pred_nn, t)
        predictions[f"prob_nn_cal_ge{t}"] = calibrated_probs(pred_nn, t, cal_nn)
    predictions.to_csv("predictions.csv", index=False)

    # bootstrap samples per position group for fig7
    def _bs_samples(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
        rng = np.random.default_rng(seed)
        n_obs = len(y_true)
        return np.array([
            mean_absolute_error(y_true[idx], y_pred[idx])
            for idx in (rng.integers(0, n_obs, n_obs) for _ in range(n))
        ])

    bs = {"overall": _bs_samples(y_test, pred_tuned)}
    for g in ["ATT", "MID", "DEF"]:
        m = (df_test["pos_group"] == g).values
        bs[g] = _bs_samples(y_test[m], pred_tuned[m])
    pd.DataFrame(bs).to_csv("bootstrap_samples.csv", index=False)

    print(f"\nmean residual: {residuals['residual'].mean():.4f}")
    print("done")


if __name__ == "__main__":
    main()
