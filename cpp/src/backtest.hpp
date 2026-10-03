// Backtest inner loop and India delivery cost model: a C++ port of
// research/backtest.py::run_backtest (NumPy loop) and costs.py::IndiaCostModel.
// Python prepares every per-rebalance array; this file does the arithmetic.
// The parity test requires the same NAV as the NumPy version to ~1e-9.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "indicators.hpp"

namespace nrl {

struct CostRates {
    double stt, exchange, sebi, stamp, gst, brokerage_rate, brokerage_cap, dp, half_spread, impact_k, scale;
};

inline double india_cost(const std::vector<double>& buy, const std::vector<double>& sell, const double* sigma,
                         const double* adv, const CostRates& r) {
    const double k = r.scale;
    double tot = 0, buy_sum = 0, brk = 0, slip = 0;
    long n_sells = 0;
    for (size_t i = 0; i < buy.size(); ++i) {
        const double to = buy[i] + sell[i];
        tot += to;
        buy_sum += buy[i];
        double b = to * r.brokerage_rate * k;
        if (r.brokerage_cap > 0) b = std::min(b, r.brokerage_cap);
        brk += b;
        if (sell[i] > 1e-6) ++n_sells;
        double s = r.half_spread * to;
        const double a = std::max(adv[i], 1.0);
        s += r.impact_k * sigma[i] * std::sqrt(to / a) * to;
        slip += s;
    }
    const double stt = r.stt * tot * k, exch = r.exchange * tot * k, sebi = r.sebi * tot * k;
    const double stamp = r.stamp * buy_sum * k;
    const double gst = r.gst * (brk + exch + sebi);
    const double dp = r.dp * n_sells * (k > 0 ? 1.0 : 0.0);
    return brk + stt + exch + sebi + stamp + gst + dp + slip * k;
}

// NumPy twin _cap_np: normalise positive weights to 1, cap each, redistribute excess pro rata.
inline std::vector<double> cap_weights(std::vector<double> w, double cap) {
    for (double& x : w) x = std::max(x, 0.0);
    double s = std::accumulate(w.begin(), w.end(), 0.0);
    if (s > 0) for (double& x : w) x /= s;
    cap = std::max(cap, 1.0 / std::max<size_t>(w.size(), 1));
    for (int it = 0; it < 50; ++it) {
        double excess = 0; bool any = false;
        std::vector<bool> over(w.size(), false);
        for (size_t i = 0; i < w.size(); ++i) if (w[i] > cap + 1e-12) { over[i] = true; any = true; }
        if (!any) break;
        for (size_t i = 0; i < w.size(); ++i) if (over[i]) { excess += w[i] - cap; w[i] = cap; }
        double room = 0;
        for (size_t i = 0; i < w.size(); ++i) if (!over[i] && w[i] > 0) room += w[i];
        if (room <= 0) break;
        for (size_t i = 0; i < w.size(); ++i) if (!over[i] && w[i] > 0) w[i] += excess * w[i] / room;
    }
    return w;
}

struct BacktestOut {
    std::vector<double> nav;                         // per day, rupees
    std::vector<double> turnover, cost, invested, value;
    std::vector<long> holdings;
    std::vector<char> traded;                        // per rebalance: 1 if the rebalance ran (>= 5 eligible)
    std::vector<double> weights;                     // K x N, fraction of portfolio after the trade
};

// R: days x N daily returns (NaN already 0); R_all: all history x N; day0: row of days[0] in R_all.
inline BacktestOut run_backtest(const double* R, long D, const double* R_all, long day0, long N,
                                const long* reb_pos, long K, const double* S, const unsigned char* OK,
                                const double* VOL, const double* SIG, const double* ADV, const double* REGM,
                                double top, int weighting, bool long_short, double max_weight, double capital,
                                double cash_d, double vol_target, double min_trade, const CostRates& rates) {
    BacktestOut o;
    o.nav.assign(D, NaN);
    o.turnover.assign(K, 0); o.cost.assign(K, 0); o.invested.assign(K, 0); o.value.assign(K, 0);
    o.holdings.assign(K, 0); o.traded.assign(K, 0); o.weights.assign(K * N, 0.0);
    std::vector<double> hold(N, 0.0), w(N), delta(N), buy(N), sell(N);
    double cash = capital;
    for (long k = 0; k < K; ++k) {
        const long j = reb_pos[k];
        double hsum = std::accumulate(hold.begin(), hold.end(), 0.0);
        const double V = cash + hsum;
        std::vector<long> elig;
        for (long i = 0; i < N; ++i) if (OK[k * N + i]) elig.push_back(i);
        if (elig.size() >= 5) {
            const long ne = static_cast<long>(elig.size());
            long n = top <= 1 ? static_cast<long>(std::nearbyint(top * ne)) : static_cast<long>(top);
            n = std::max(1L, std::min(n, ne));
            std::vector<long> order = elig;
            std::stable_sort(order.begin(), order.end(), [&](long a, long b) { return S[k * N + a] > S[k * N + b]; });
            std::vector<double> raw(n);
            if (weighting == 1) {                                      // score-weighted
                double mn = S[k * N + elig[0]];
                for (long i : elig) mn = std::min(mn, S[k * N + i]);
                for (long q = 0; q < n; ++q) raw[q] = std::max(S[k * N + order[q]] - mn, 1e-9);
            } else if (weighting == 2) {                               // inverse volatility
                std::vector<double> vv;
                for (long i : elig) if (ok(VOL[k * N + i])) vv.push_back(VOL[k * N + i]);
                double med = NaN;
                if (!vv.empty()) {
                    std::sort(vv.begin(), vv.end());
                    const size_t m = vv.size();
                    med = m % 2 ? vv[m / 2] : 0.5 * (vv[m / 2 - 1] + vv[m / 2]);
                }
                for (long q = 0; q < n; ++q) {
                    double v = VOL[k * N + order[q]];
                    if (!(ok(v) && v > 0)) v = med;
                    raw[q] = 1.0 / v;
                }
            } else {
                std::fill(raw.begin(), raw.end(), 1.0);
            }
            std::fill(w.begin(), w.end(), 0.0);
            auto wl = cap_weights(raw, max_weight);
            for (long q = 0; q < n; ++q) w[order[q]] = wl[q];
            if (long_short) {
                auto ws = cap_weights(std::vector<double>(n, 1.0), max_weight);
                for (long q = 0; q < n; ++q) w[order[ne - n + q]] -= ws[q];
            }
            double invested = REGM[k];
            if (vol_target > 0) {
                const long g = day0 + j, a = std::max(0L, g - 62);
                const long m = g - a + 1;
                std::vector<double> pr(m, 0.0);
                for (long t = a; t <= g; ++t) {
                    double s = 0;
                    for (long i = 0; i < N; ++i) s += R_all[t * N + i] * w[i];
                    pr[t - a] = s;
                }
                const double mean = std::accumulate(pr.begin(), pr.end(), 0.0) / m;
                double ss = 0;
                for (double x : pr) ss += (x - mean) * (x - mean);
                const double pv = std::sqrt(ss / m) * std::sqrt(252.0);
                if (pv > 0) invested *= std::min(1.0, vol_target / pv);
            }
            double V_eff = V, c = 0;
            for (int it = 0; it < 3; ++it) {
                for (long i = 0; i < N; ++i) {
                    double d = w[i] * invested * V_eff - hold[i];
                    if (std::fabs(d) < min_trade * V) d = 0.0;
                    delta[i] = d; buy[i] = std::max(d, 0.0); sell[i] = std::max(-d, 0.0);
                }
                c = india_cost(buy, sell, SIG + k * N, ADV + k * N, rates);
                V_eff = V - c;
            }
            double after = c - V, buy_sum = 0;
            for (long i = 0; i < N; ++i) { after += hold[i] + delta[i]; buy_sum += buy[i]; }
            if (!long_short && after > 0 && buy_sum > 0) {
                const double f = std::max(0.0, 1 - after / buy_sum);
                for (long i = 0; i < N; ++i) if (delta[i] > 0) { delta[i] *= f; buy[i] = delta[i]; }
                c = india_cost(buy, sell, SIG + k * N, ADV + k * N, rates);
            }
            double absd = 0;
            for (long i = 0; i < N; ++i) { absd += std::fabs(delta[i]); hold[i] += delta[i]; }
            hsum = std::accumulate(hold.begin(), hold.end(), 0.0);
            cash = V - hsum - c;
            long nz = 0;
            for (long i = 0; i < N; ++i) {
                if (hold[i] != 0) { ++nz; o.weights[k * N + i] = hold[i] / (cash + hsum); }
            }
            o.turnover[k] = absd / V / (long_short ? 4.0 : 2.0);
            o.cost[k] = c; o.holdings[k] = nz; o.invested[k] = invested; o.value[k] = V; o.traded[k] = 1;
        }
        hsum = std::accumulate(hold.begin(), hold.end(), 0.0);
        o.nav[j] = cash + hsum;
        const long j1 = (k + 1 < K) ? reb_pos[k + 1] : D - 1;
        if (j1 > j) {
            std::vector<double> H = hold;
            double cg = cash;
            for (long t = j + 1; t <= j1; ++t) {
                double s = 0;
                for (long i = 0; i < N; ++i) { H[i] *= 1 + R[t * N + i]; s += H[i]; }
                cg *= 1 + cash_d;
                o.nav[t] = cg + s;
            }
            hold = H;
            cash = cg;
        }
    }
    return o;
}

}  // namespace nrl
