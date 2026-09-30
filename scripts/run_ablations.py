"""Queue every ablation × seed, then aggregate validation scores.

    python scripts/run_ablations.py --steps 300000 --seeds 0 1 2 3 4     # train (hours; resumable)
    python scripts/run_ablations.py --aggregate-only                      # tables + bar charts

Runs are skipped if ``runs/<tag>/last.pt`` (or ``last.zip`` for SB3) already exists,
so the queue can be interrupted and restarted. Keep the machine awake while it runs.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from niveshrl import plots
from niveshrl.algos.ppo import read_log

# name -> (script, extra args, run-tag template)
VARIANTS = {
    "PPO + CNN + attention (full)": ("train_custom.py", [], "ppo_cnn_utility_reb5_s{s}"),
    "PPO + LSTM + attention": ("train_custom.py", ["--temporal", "lstm"], "ppo_lstm_utility_reb5_s{s}"),
    "PPO + MLP, no attention": ("train_custom.py", ["--temporal", "mlp", "--no-attention"], "ppo_mlp_noattn_utility_reb5_s{s}"),
    "A2C + CNN + attention": ("train_custom.py", ["--algo", "a2c"], "a2c_cnn_utility_reb5_s{s}"),
    "PPO, DSR reward (profile-blind)": ("train_custom.py", ["--reward", "dsr"], "ppo_cnn_dsr_reb5_s{s}"),
    "PPO, Sortino reward": ("train_custom.py", ["--reward", "sortino"], "ppo_cnn_sortino_reb5_s{s}"),
    "PPO, penalised-return reward": ("train_custom.py", ["--reward", "penalized"], "ppo_cnn_penalized_reb5_s{s}"),
    "PPO, daily rebalance": ("train_custom.py", ["--rebalance", "1"], "ppo_cnn_utility_reb1_s{s}"),
    "PPO, tax-aware": ("train_custom.py", ["--tax-aware"], "ppo_cnn_utility_reb5_tax_s{s}"),
    "SB3 PPO (MLP)": ("train_sb3.py", ["--algo", "ppo"], "sb3_ppo_mlp_s{s}"),
    "SB3 PPO (our encoder)": ("train_sb3.py", ["--algo", "ppo", "--custom-encoder"], "sb3_ppo_enc_s{s}"),
    "SB3 SAC (MLP)": ("train_sb3.py", ["--algo", "sac"], "sb3_sac_mlp_s{s}"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300_000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--only", nargs="*", help="substring filter on variant names")
ap.add_argument("--aggregate-only", action="store_true")
args = ap.parse_args()

selected = {k: v for k, v in VARIANTS.items() if not args.only or any(o.lower() in k.lower() for o in args.only)}

if not args.aggregate_only:
    for name, (script, extra, tag_t) in selected.items():
        for s in args.seeds:
            run = Path("runs") / tag_t.format(s=s)
            if (run / "last.pt").exists() or (run / "last.zip").exists():
                print(f"skip (done): {run}")
                continue
            cmd = [sys.executable, f"scripts/{script}", "--steps", str(args.steps), "--seed", str(s), *extra]
            if script == "train_custom.py":
                cmd += ["--out", run.name]
            print(f"\n=== {name} | seed {s} ===\n{' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)


def best_val(run: Path) -> float | None:
    log = run / "log.csv"
    if log.exists():  # custom PPO
        df = read_log(log)
        return float(df["val_Sharpe"].max()) if "val_Sharpe" in df else None  # comparable across rewards
    ev = run / "val_scores.csv"  # SB3 runs write this from their callback
    return float(pd.read_csv(ev)["Sharpe"].max()) if ev.exists() else None


rows, logs = [], {}
for name, (_, _, tag_t) in selected.items():
    for s in args.seeds:
        run = Path("runs") / tag_t.format(s=s)
        v = best_val(run)
        if v is not None:
            rows.append({"variant": name, "seed": s, "val_sharpe": v})
            if s == args.seeds[0] and (run / "log.csv").exists():
                logs[name] = read_log(run / "log.csv")
if not rows:
    sys.exit("no finished runs to aggregate")
df = pd.DataFrame(rows)
out = Path("report/results")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "ablations_val.csv", index=False)
agg = df.groupby("variant")["val_sharpe"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
agg["std"] = agg["std"].fillna(0.0)
print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
fig_dir = Path("report/figures/val")
fig_dir.mkdir(parents=True, exist_ok=True)
plots.bar_with_errors(agg, fig_dir / "ablations.png", "Ablations: best validation Sharpe (mean ± std over seeds)")
if logs:
    plots.training_curves(logs, fig_dir / "training_curves.png")
