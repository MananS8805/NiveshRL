"""Project portfolio weights onto the feasible set.

The feasible set is:
- long-only, summing to 1 (index N is cash)
- each stock <= ``max_stock``
- each sector <= ``max_sector``
- cash >= ``min_cash``

Weight above a cap is redistributed in proportion to the entries that still
have room. Cash is uncapped, so a feasible point always exists and the loop
terminates. Because the env applies this projection before every trade, the
limits hold no matter what the policy outputs.
"""
from __future__ import annotations

import numpy as np


def project(w: np.ndarray, sector_ids: np.ndarray, max_stock: float = 0.10,
            max_sector: float = 0.30, min_cash: float = 0.0, iters: int = 100) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.eye(len(w))[-1]
    n = len(w) - 1
    n_sec = int(sector_ids.max()) + 1

    # Stocks never get *more* than 1 - min_cash in total.
    stock_budget = 1.0 - min_cash
    for _ in range(iters):
        stocks = w[:n]
        capped = np.minimum(stocks, max_stock)
        sec_tot = np.bincount(sector_ids, weights=capped, minlength=n_sec)
        scale = np.where(sec_tot > max_sector, max_sector / np.maximum(sec_tot, 1e-12), 1.0)
        capped = capped * scale[sector_ids]
        excess = stocks.sum() - capped.sum()
        w = np.append(capped, w[n])
        if excess <= 1e-12:
            break
        # Room left for each stock under both its own cap and its sector cap.
        sec_tot = np.bincount(sector_ids, weights=capped, minlength=n_sec)
        room_stock = np.maximum(max_stock - capped, 0.0)
        room_sec = np.maximum(max_sector - sec_tot, 0.0)[sector_ids]
        has_room = (room_stock > 1e-9) & (room_sec > 1e-9) & (capped > 0)
        recv = np.append(np.where(has_room, capped, 0.0), w[n])  # cash always receives
        if recv.sum() <= 1e-12:
            w[n] += excess
            break
        w = w + excess * recv / recv.sum()

    # Hard final pass: anything still above a cap goes to cash (guarantees feasibility).
    capped = np.minimum(w[:n], max_stock)
    sec_tot = np.bincount(sector_ids, weights=capped, minlength=n_sec)
    capped *= np.where(sec_tot > max_sector, max_sector / np.maximum(sec_tot, 1e-12), 1.0)[sector_ids]
    w = np.append(capped, 1.0 - capped.sum())

    if w[:n].sum() > stock_budget:
        w[:n] *= stock_budget / w[:n].sum()
        w[n] = 1.0 - w[:n].sum()
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def is_feasible(w: np.ndarray, sector_ids: np.ndarray, max_stock: float, max_sector: float,
                min_cash: float, tol: float = 1e-6) -> bool:
    n = len(w) - 1
    sec = np.bincount(sector_ids, weights=w[:n])
    return bool(np.all(w >= -tol) and abs(w.sum() - 1) < tol and np.all(w[:n] <= max_stock + tol)
                and np.all(sec <= max_sector + tol) and w[n] >= min_cash - tol)
