"""One-command demo: prepare whatever is missing, then open the dashboard.

    python scripts/demo.py            # quick demo (fast models only), then launch the app
    python scripts/demo.py --full     # also train the slow models (LSTM/Transformer rankers, vol LSTM)
    python scripts/demo.py --no-launch

Every step is skipped if its output already exists, so rerunning is cheap.
Called by demo.bat (double-click on Windows).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DATA = ROOT / "data"
PRED = DATA / "predictions"
NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"


def step(title: str, cmd: list[str] | None = None, fn=None) -> None:
    print(f"\n=== {title} ===", flush=True)
    t = time.time()
    if cmd:
        subprocess.run([PY, *cmd], cwd=ROOT, check=True)
    elif fn:
        fn()
    print(f"    done in {time.time() - t:.0f}s", flush=True)


def skip(title: str, why: str) -> None:
    print(f"--- {title}: skipped ({why})", flush=True)


def download_universe() -> None:
    req = urllib.request.Request(NIFTY200_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        (DATA / "ind_nifty200list.csv").write_bytes(r.read())


def has_rl_model() -> bool:
    return any((p / "args.json").exists() for p in (ROOT / "runs").glob("*/") if (p / "best.pt").exists())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also train the slow models (hours on CPU)")
    ap.add_argument("--no-launch", action="store_true", help="prepare everything but don't start the app")
    ap.add_argument("--port", default="8501")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    # 1. Data
    if (DATA / "ind_nifty200list.csv").exists():
        skip("NIFTY 200 constituent list", "present")
    else:
        step("Download NIFTY 200 constituent list from NSE", fn=download_universe)
    if (DATA / "stocks.parquet").exists():
        skip("RL data (29 NIFTY 50 stocks)", "cached")
    else:
        step("Download + clean RL data (29 NIFTY 50 stocks)", ["scripts/prepare_data.py"])

    # 2. Research models (the NIFTY 200 panel downloads automatically on first use)
    fast = ["momentum", "logreg", "ffnn"]
    slow = ["lstm", "transformer"] if args.full else []
    missing = [m for m in fast + slow if not (PRED / f"{m}.parquet").exists()]
    if missing:
        step(f"Walk-forward train stock rankers: {', '.join(missing)}",
             ["scripts/train_rankers.py", "--models", *missing])
    else:
        skip("Stock rankers", "predictions present")
    if (PRED / "regimes.parquet").exists():
        skip("Regime detector", "present")
    else:
        step("Walk-forward regime detector", ["scripts/train_regimes.py"])
    if args.full and not (PRED / "vol_forecasts.parquet").exists():
        step("Volatility forecaster (LSTM vs GARCH) - slow", ["scripts/train_volatility.py"])

    # 3. RL allocator (RL + PLAN screens)
    if has_rl_model():
        skip("RL allocator", "a trained checkpoint exists")
    else:
        steps = "100000" if args.full else "20000"
        step(f"Train RL allocator ({steps} steps; a demo-size run)",
             ["scripts/train_custom.py", "--steps", steps, "--eval-every", "10000", "--out", "demo_rl"])

    if args.no_launch:
        print("\nEverything is ready. Start the app with:  python -m streamlit run app.py")
        return
    print(f"\n=== Launching NiveshRL terminal on http://localhost:{args.port} (Ctrl+C to stop) ===", flush=True)
    subprocess.run([PY, "-m", "streamlit", "run", "app.py", "--server.port", args.port,
                    "--server.headless", "false"], cwd=ROOT)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nA step failed ({' '.join(map(str, e.cmd[1:]))}). See the messages above.")
    except KeyboardInterrupt:
        print("\nStopped.")
