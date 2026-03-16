# src/backtest_regime_direction.py
"""
Backtest: Does knowing the regime help predict BTC 15-min direction?

For each 5-min candle in the labeled dataset:
1. Look at the regime at time T
2. Determine the directional signal (momentum indicators)
3. Check actual price 15 min later (3 candles forward)
4. Score: did the signal + regime filter beat 50/50?

Usage:
    cd CryptoMarket_Regime_Classifier
    python -m src.backtest_regime_direction
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_FILE = "data/BTCUSDT_combined_klines_20230316_20260315_states6_labeled.csv"
FORWARD_BARS = 3  # 3 x 5min = 15 minutes


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    # forward return: price change 15 min from now
    df["fwd_ret_15m"] = df["close_5m"].shift(-FORWARD_BARS) / df["close_5m"] - 1
    df["fwd_up"] = (df["fwd_ret_15m"] > 0).astype(int)
    df = df.dropna(subset=["fwd_ret_15m", "regime"])
    return df


def directional_signal(df):
    """
    Simple momentum signal using features already in the data:
    - log_ret_1_5m > 0 (recent return positive)
    - ema_ratio_9_21_5m > 0 (short EMA above long EMA)
    - macd_hist_5m > 0 (MACD momentum positive)

    Signal = majority vote of the 3. Returns 1 (UP) or 0 (DOWN).
    """
    votes = pd.DataFrame()
    votes["ret"] = (df["log_ret_1_5m"] > 0).astype(int)
    votes["ema"] = (df["ema_ratio_9_21_5m"] > 0).astype(int)
    votes["macd"] = (df["macd_hist_5m"] > 0).astype(int)
    df = df.copy()
    df["signal"] = (votes.sum(axis=1) >= 2).astype(int)  # majority = UP
    return df


def backtest(df):
    print(f"\nTotal samples: {len(df):,}")
    print(f"Baseline (always guess UP): {df['fwd_up'].mean():.2%}")
    print(f"{'='*70}\n")

    # --- Overall directional accuracy (no regime filter) ---
    correct_no_filter = (df["signal"] == df["fwd_up"]).mean()
    print(f"Direction signal accuracy (all regimes): {correct_no_filter:.2%}")
    print()

    # --- Per-regime breakdown ---
    print(f"{'Regime':<20s} {'Count':>8s} {'%Data':>7s} {'DirAcc':>8s} {'AvgRet':>10s} {'Edge':>8s} {'Actionable?'}")
    print(f"{'-'*85}")

    results = []
    for regime in sorted(df["regime"].unique()):
        mask = df["regime"] == regime
        sub = df[mask]
        n = len(sub)
        pct = n / len(df)

        # directional accuracy within this regime
        dir_acc = (sub["signal"] == sub["fwd_up"]).mean()

        # average signed return when following the signal
        # if signal=UP, return = fwd_ret; if signal=DOWN, return = -fwd_ret
        signed_ret = np.where(sub["signal"] == 1, sub["fwd_ret_15m"], -sub["fwd_ret_15m"])
        avg_ret = signed_ret.mean()

        # edge over 50%
        edge = dir_acc - 0.5

        actionable = "YES" if abs(edge) > 0.02 else "maybe" if abs(edge) > 0.01 else "no"

        results.append({
            "regime": regime, "n": n, "pct": pct,
            "dir_acc": dir_acc, "avg_ret": avg_ret, "edge": edge,
        })
        print(f"{regime:<20s} {n:>8,d} {pct:>7.1%} {dir_acc:>8.2%} {avg_ret:>+10.5%} {edge:>+8.2%}   {actionable}")

    print(f"\n{'='*70}")

    # --- Strategy: only bet when regime is "actionable" ---
    res_df = pd.DataFrame(results)
    good_regimes = res_df[res_df["edge"].abs() > 0.02]["regime"].tolist()

    if good_regimes:
        print(f"\nActionable regimes (edge > 2%): {good_regimes}")
        actionable_mask = df["regime"].isin(good_regimes)
        sub = df[actionable_mask]
        acc = (sub["signal"] == sub["fwd_up"]).mean()
        signed = np.where(sub["signal"] == 1, sub["fwd_ret_15m"], -sub["fwd_ret_15m"])
        print(f"Filtered accuracy: {acc:.2%} (on {len(sub):,} samples = {len(sub)/len(df):.1%} of data)")
        print(f"Avg return per trade: {signed.mean():+.5%}")
        print(f"Cumulative return: {signed.sum():+.2%}")
    else:
        print("\nNo regime showed > 2% directional edge.")

    # --- Regime transition analysis ---
    print(f"\n{'='*70}")
    print("Regime Transition → 15min Direction")
    print(f"{'='*70}")
    print("\nWhen regime CHANGES, what happens to price in the next 15 min?\n")

    df_copy = df.copy()
    df_copy["prev_regime"] = df_copy["regime"].shift(1)
    df_copy["regime_changed"] = df_copy["regime"] != df_copy["prev_regime"]
    transitions = df_copy[df_copy["regime_changed"]].copy()

    if len(transitions) > 100:
        print(f"{'Transition':<40s} {'Count':>6s} {'UpPct':>7s} {'AvgRet':>10s}")
        print(f"{'-'*70}")
        for (prev, curr), grp in transitions.groupby(["prev_regime", "regime"]):
            if len(grp) >= 20:
                up_pct = grp["fwd_up"].mean()
                avg_ret = grp["fwd_ret_15m"].mean()
                label = f"{prev} -> {curr}"
                print(f"{label:<40s} {len(grp):>6d} {up_pct:>7.1%} {avg_ret:>+10.5%}")


if __name__ == "__main__":
    print("Loading labeled data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows with 15-min forward returns")
    df = directional_signal(df)
    backtest(df)
