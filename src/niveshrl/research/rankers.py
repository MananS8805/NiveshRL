"""Deep-learning stock rankers and a rolling-window walk-forward trainer.

Models predict P(next-month return > cross-sectional median) and are then
used to *rank* stocks each month:

- ``ffnn``: Takeuchi & Lee (2013) feed-forward net, 33-40-4-50-1, with a
  4-unit bottleneck. The paper pretrained it as stacked RBMs; here it is
  trained end-to-end with dropout and early stopping (no pretraining).
- ``lstm``: LSTM over the 13 monthly returns, concatenated with the 20
  daily features and the January dummy.
- ``transformer``: 2-layer Transformer encoder over the 13 monthly tokens,
  with the same static context.
- ``logreg``: logistic regression on the same 33 features (linear baseline).
- ``momentum``: classic 12-1 momentum score (feature ``m12``), no training.

**Walk-forward, rolling window.** For each test year Y, the models train on
months in ``[Y - window, Y - 1]``. The last 12 of those months are held out
for early stopping, and the models then predict every month of Y. A
training sample dated t has its label realised at the end of month t+1,
which is at or before the first test date, so no future information
enters. Hyperparameters are fixed up front (the paper's architecture,
standard optimiser settings), not tuned on the out-of-sample years.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import ROOT
from .features import FEATURES, SEQ_LEN, STATIC, RankData

PRED_DIR = ROOT / "data" / "predictions"
DL_MODELS = ["ffnn", "lstm", "transformer"]
ALL_MODELS = DL_MODELS + ["logreg", "momentum"]


class FFNN(nn.Module):
    """33-40-4-50-1 with a bottleneck (Takeuchi & Lee, 2013)."""

    def __init__(self, n_in: int = len(FEATURES), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 40), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(40, 4), nn.ReLU(),                      # bottleneck
            nn.Linear(4, 50), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(50, 1),
        )

    def forward(self, x, static=None):
        return self.net(x).squeeze(-1)


class LSTMRanker(nn.Module):
    def __init__(self, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + len(STATIC), 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], -1)).squeeze(-1)


class TransformerRanker(nn.Module):
    def __init__(self, d: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d + len(STATIC), 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def forward(self, seq, static):
        h = self.enc(self.embed(seq) + self.pos)
        return self.head(torch.cat([h.mean(1), static], -1)).squeeze(-1)


def make_model(name: str) -> nn.Module:
    return {"ffnn": FFNN, "lstm": LSTMRanker, "transformer": TransformerRanker}[name]()


def _tensors(rd: RankData, rows: pd.DataFrame, name: str):
    if name == "ffnn":
        return (torch.from_numpy(rd.X(rows)), torch.zeros(len(rows), 0))
    s, st = rd.seq(rows)
    return torch.from_numpy(s), torch.from_numpy(st)


@dataclass
class TrainConfig:
    window_years: int = 8
    val_months: int = 12
    epochs: int = 60
    patience: int = 6
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


def fit_torch(name: str, rd: RankData, train: pd.DataFrame, val: pd.DataFrame, cfg: TrainConfig) -> nn.Module:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    m = make_model(name)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    a_tr, b_tr = _tensors(rd, train, name)
    y_tr = torch.tensor(train["label"].to_numpy(np.float32))
    a_va, b_va = _tensors(rd, val, name)
    y_va = torch.tensor(val["label"].to_numpy(np.float32))
    best, best_state, bad = np.inf, None, 0
    n = len(y_tr)
    for _ in range(cfg.epochs):
        m.train()
        perm = torch.randperm(n)
        for s in range(0, n, cfg.batch):
            i = perm[s:s + cfg.batch]
            loss = lossf(m(a_tr[i], b_tr[i]), y_tr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(a_va, b_va), y_va).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    m.load_state_dict(best_state)
    return m.eval()


def predict_torch(m: nn.Module, name: str, rd: RankData, rows: pd.DataFrame) -> np.ndarray:
    a, b = _tensors(rd, rows, name)
    with torch.no_grad():
        return torch.sigmoid(m(a, b)).numpy()


def walk_forward(rd: RankData, name: str, first_test_year: int = 2012, cfg: TrainConfig | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Out-of-sample scores for every month from ``first_test_year`` on.

    Returns a frame indexed (date, ticker) with columns score, ret_next, label.
    """
    cfg = cfg or TrainConfig()
    f = rd.frame
    dates = f.index.get_level_values(0)
    years = sorted({d.year for d in dates if d.year >= first_test_year})
    out = []
    for Y in years:
        test = f[dates.year == Y]
        if name == "momentum":
            score = test["m12"].to_numpy()
        else:
            lo = pd.Timestamp(f"{Y - cfg.window_years}-01-01")
            hist = f[(dates >= lo) & (dates < pd.Timestamp(f"{Y}-01-01"))].dropna(subset=["label"])
            hd = hist.index.get_level_values(0)
            cut = hd.unique().sort_values()[-cfg.val_months]
            train, val = hist[hd < cut], hist[hd >= cut]
            if name == "logreg":
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1.0, max_iter=500).fit(rd.X(train), train["label"])
                score = lr.predict_proba(rd.X(test))[:, 1]
            else:
                model = fit_torch(name, rd, train, val, cfg)
                score = predict_torch(model, name, rd, test)
        out.append(pd.DataFrame({"score": score, "ret_next": test["ret_next"].to_numpy(),
                                 "label": test["label"].to_numpy()}, index=test.index))
        if verbose:
            print(f"  [{name}] {Y}: {len(test)} stock-months", flush=True)
    return pd.concat(out)


def save_predictions(name: str, pred: pd.DataFrame) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{name}.parquet"
    pred.to_parquet(path)
    return path


def load_predictions(name: str) -> pd.DataFrame | None:
    path = PRED_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None
