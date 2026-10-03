// Technical indicators over (T x N) column-major-by-stock arrays.
//
// These are exact ports of the pandas operations used in
// src/niveshrl/research/technicals.py (the reference implementation), including
// pandas' NaN semantics, so the parity tests can require agreement to ~1e-9:
//   - ewm(alpha, adjust=False, min_periods, ignore_na=False)
//   - rolling(n, min_periods).mean() / .std(ddof) / .max() / .min()
//   - the Supertrend recursion
// Arrays are passed as row-major T x N doubles (pandas .to_numpy() order).
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace nrl {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
inline bool ok(double x) { return !std::isnan(x); }

// pandas ewm(alpha, adjust=False, ignore_na=False).mean() with min_periods, one column.
inline void ewm_col(const double* x, double* out, long T, long N, long j, double alpha, long minp) {
    const double owf = 1.0 - alpha, nw = alpha;
    double weighted = x[j];
    double old_wt = 1.0;
    long nobs = ok(weighted) ? 1 : 0;
    out[j] = (nobs >= minp) ? weighted : NaN;
    for (long t = 1; t < T; ++t) {
        const double cur = x[t * N + j];
        const bool is_obs = ok(cur);
        nobs += is_obs;
        if (ok(weighted)) {
            old_wt *= owf;                       // ignore_na=False: decay even across gaps
            if (is_obs) {
                if (weighted != cur) {
                    weighted = old_wt * weighted + nw * cur;
                    weighted /= (old_wt + nw);
                }
                old_wt = 1.0;                    // adjust=False
            }
        } else if (is_obs) {
            weighted = cur;
        }
        out[t * N + j] = (nobs >= minp) ? weighted : NaN;
    }
}

inline void ewm(const double* x, double* out, long T, long N, double alpha, long minp) {
    for (long j = 0; j < N; ++j) ewm_col(x, out, T, N, j, alpha, minp);
}

// pandas rolling(n, min_periods).mean(), NaN-aware.
inline void rolling_mean(const double* x, double* out, long T, long N, long n, long minp) {
    for (long j = 0; j < N; ++j) {
        double s = 0; long c = 0;
        for (long t = 0; t < T; ++t) {
            const double v = x[t * N + j];
            if (ok(v)) { s += v; ++c; }
            if (t >= n) { const double o = x[(t - n) * N + j]; if (ok(o)) { s -= o; --c; } }
            out[t * N + j] = (c >= minp && c > 0) ? s / c : NaN;
        }
    }
}

// pandas rolling(n, min_periods).std(ddof), recomputed per window (two-pass, numerically stable).
inline void rolling_std(const double* x, double* out, long T, long N, long n, long minp, int ddof) {
    for (long j = 0; j < N; ++j) {
        for (long t = 0; t < T; ++t) {
            const long a = std::max(0L, t - n + 1);
            double s = 0; long c = 0;
            for (long k = a; k <= t; ++k) { const double v = x[k * N + j]; if (ok(v)) { s += v; ++c; } }
            if (c < minp || c - ddof <= 0) { out[t * N + j] = NaN; continue; }
            const double m = s / c;
            double ss = 0;
            for (long k = a; k <= t; ++k) { const double v = x[k * N + j]; if (ok(v)) ss += (v - m) * (v - m); }
            out[t * N + j] = std::sqrt(ss / (c - ddof));
        }
    }
}

// pandas rolling(n, min_periods).max() / .min() via a monotonic deque: O(T) per column.
inline void rolling_extreme(const double* x, double* out, long T, long N, long n, long minp, bool is_max) {
    for (long j = 0; j < N; ++j) {
        std::deque<long> dq;
        long c = 0;
        for (long t = 0; t < T; ++t) {
            const double v = x[t * N + j];
            if (ok(v)) {
                ++c;
                while (!dq.empty()) {
                    const double b = x[dq.back() * N + j];
                    if (is_max ? (b <= v) : (b >= v)) dq.pop_back(); else break;
                }
                dq.push_back(t);
            }
            if (t >= n && ok(x[(t - n) * N + j])) --c;
            while (!dq.empty() && dq.front() <= t - n) dq.pop_front();
            out[t * N + j] = (c >= minp && !dq.empty()) ? x[dq.front() * N + j] : NaN;
        }
    }
}

// Wilder RSI exactly as technicals.rsi(): ewm(alpha=1/n, adjust=False, min_periods=n) of gains/losses.
inline void rsi(const double* c, double* out, long T, long N, long n) {
    std::vector<double> up(T * N), dn(T * N), eu(T * N), ed(T * N);
    for (long j = 0; j < N; ++j) {
        up[j] = dn[j] = NaN;
        for (long t = 1; t < T; ++t) {
            const double a = c[t * N + j], b = c[(t - 1) * N + j];
            const double d = (ok(a) && ok(b)) ? a - b : NaN;
            up[t * N + j] = ok(d) ? std::max(d, 0.0) : NaN;
            dn[t * N + j] = ok(d) ? std::max(-d, 0.0) : NaN;
        }
    }
    ewm(up.data(), eu.data(), T, N, 1.0 / n, n);
    ewm(dn.data(), ed.data(), T, N, 1.0 / n, n);
    for (long i = 0; i < T * N; ++i) {
        if (!ok(c[i])) { out[i] = NaN; continue; }
        if (ok(ed[i]) && ed[i] == 0.0) { out[i] = 100.0; continue; }
        out[i] = (ok(eu[i]) && ok(ed[i])) ? 100.0 - 100.0 / (1.0 + eu[i] / ed[i]) : NaN;
    }
}

// True range with pandas' max-of-available semantics (technicals.true_range).
inline void true_range(const double* h, const double* l, const double* c, double* out, long T, long N) {
    for (long j = 0; j < N; ++j) for (long t = 0; t < T; ++t) {
        const long i = t * N + j;
        const double prev = t > 0 ? c[(t - 1) * N + j] : NaN;
        double m = NaN;
        const double cand[3] = {h[i] - l[i], std::fabs(h[i] - prev), std::fabs(l[i] - prev)};
        for (double v : cand) if (ok(v)) m = ok(m) ? std::max(m, v) : v;
        out[i] = m;
    }
}

// Supertrend direction (+1 / -1), the same recursion as technicals.supertrend_dir.
inline void supertrend(const double* hl2, const double* atr_, const double* c, double* out, long T, long N, double mult) {
    for (long j = 0; j < N; ++j) {
        double fu = NaN, fd = NaN; int d = 1;
        for (long t = 0; t < T; ++t) {
            const long i = t * N + j;
            out[i] = NaN;
            const double a = atr_[i];
            if (!(ok(a) && ok(hl2[i]) && ok(c[i]))) continue;
            const double nu = hl2[i] - mult * a, nd = hl2[i] + mult * a;
            const double pc = t > 0 ? c[(t - 1) * N + j] : NaN;
            if (ok(fu) && ok(pc)) {
                fu = (pc > fu) ? std::max(nu, fu) : nu;
                fd = (pc < fd) ? std::min(nd, fd) : nd;
            } else { fu = nu; fd = nd; }
            if (c[i] > fd) d = 1; else if (c[i] < fu) d = -1;
            out[i] = d;
        }
    }
}

}  // namespace nrl
