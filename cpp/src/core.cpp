// niveshrl_core: pybind11 bindings for the C++ hot paths.
//   indicators: ewm, rolling_mean/std/max/min, rsi, true_range, supertrend
//   screener:   apply_filters over a column matrix
//   backtest:   run_backtest (port of the NumPy loop, with India costs)
//   ticks:      TickStore (live quotes + 1-minute bars, fixed memory)
// Heavy calls release the GIL so the Qt UI thread and the feed thread keep running.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backtest.hpp"
#include "indicators.hpp"
#include "ticks.hpp"

namespace py = pybind11;
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::pair<long, long> shape2(const arr& a) {
    if (a.ndim() != 2) throw std::runtime_error("expected a 2-D (T x N) array");
    return {static_cast<long>(a.shape(0)), static_cast<long>(a.shape(1))};
}

template <typename F>
static arr unary(const arr& x, F f) {
    auto [T, N] = shape2(x);
    arr out({T, N});
    const double* xp = x.data();
    double* op = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        f(xp, op, T, N);
    }
    return out;
}

PYBIND11_MODULE(niveshrl_core, m) {
    m.doc() = "NiveshRL C++ core: indicators, screener, backtest loop, live tick store";

    m.def("ewm", [](const arr& x, double alpha, long minp) {
        return unary(x, [&](const double* a, double* o, long T, long N) { nrl::ewm(a, o, T, N, alpha, minp); });
    }, py::arg("x"), py::arg("alpha"), py::arg("min_periods") = 0,
          "pandas ewm(alpha, adjust=False, ignore_na=False).mean() column-wise");
    m.def("rolling_mean", [](const arr& x, long n, long minp) {
        return unary(x, [&](const double* a, double* o, long T, long N) { nrl::rolling_mean(a, o, T, N, n, minp); });
    }, py::arg("x"), py::arg("window"), py::arg("min_periods"));
    m.def("rolling_std", [](const arr& x, long n, long minp, int ddof) {
        return unary(x, [&](const double* a, double* o, long T, long N) { nrl::rolling_std(a, o, T, N, n, minp, ddof); });
    }, py::arg("x"), py::arg("window"), py::arg("min_periods"), py::arg("ddof") = 1);
    m.def("rolling_max", [](const arr& x, long n, long minp) {
        return unary(x, [&](const double* a, double* o, long T, long N) { nrl::rolling_extreme(a, o, T, N, n, minp, true); });
    }, py::arg("x"), py::arg("window"), py::arg("min_periods"));
    m.def("rolling_min", [](const arr& x, long n, long minp) {
        return unary(x, [&](const double* a, double* o, long T, long N) { nrl::rolling_extreme(a, o, T, N, n, minp, false); });
    }, py::arg("x"), py::arg("window"), py::arg("min_periods"));
    m.def("rsi", [](const arr& c, long n) {
        return unary(c, [&](const double* a, double* o, long T, long N) { nrl::rsi(a, o, T, N, n); });
    }, py::arg("close"), py::arg("n") = 14);
    m.def("true_range", [](const arr& h, const arr& l, const arr& c) {
        auto [T, N] = shape2(c);
        arr out({T, N});
        { py::gil_scoped_release nogil; nrl::true_range(h.data(), l.data(), c.data(), out.mutable_data(), T, N); }
        return out;
    });
    m.def("supertrend", [](const arr& hl2, const arr& atr, const arr& c, double mult) {
        auto [T, N] = shape2(c);
        arr out({T, N});
        { py::gil_scoped_release nogil; nrl::supertrend(hl2.data(), atr.data(), c.data(), out.mutable_data(), T, N, mult); }
        return out;
    }, py::arg("hl2"), py::arg("atr"), py::arg("close"), py::arg("mult") = 3.0);

    // Screener: X is (N stocks x C columns); filters are (column index, op code, value);
    // op codes 0 >, 1 >=, 2 <, 3 <=, 4 ==. NaN never passes. Returns a bool mask.
    m.def("apply_filters", [](const arr& X, const std::vector<std::tuple<int, int, double>>& filters) {
        auto [N, C] = shape2(X);
        py::array_t<bool> out(N);
        bool* o = out.mutable_data();
        const double* x = X.data();
        {
            py::gil_scoped_release nogil;
            for (long i = 0; i < N; ++i) {
                bool keep = true;
                for (auto& [col, op, val] : filters) {
                    const double v = x[i * C + col];
                    if (std::isnan(v)) { keep = false; break; }
                    bool pass = op == 0 ? v > val : op == 1 ? v >= val : op == 2 ? v < val : op == 3 ? v <= val : v == val;
                    if (!pass) { keep = false; break; }
                }
                o[i] = keep;
            }
        }
        return out;
    });

    py::class_<nrl::CostRates>(m, "CostRates")
        .def(py::init([](double stt, double exchange, double sebi, double stamp, double gst, double brokerage_rate,
                         double brokerage_cap, double dp, double half_spread, double impact_k, double scale) {
            return nrl::CostRates{stt, exchange, sebi, stamp, gst, brokerage_rate, brokerage_cap, dp, half_spread,
                                  impact_k, scale};
        }), py::arg("stt"), py::arg("exchange"), py::arg("sebi"), py::arg("stamp"), py::arg("gst"),
             py::arg("brokerage_rate"), py::arg("brokerage_cap"), py::arg("dp"), py::arg("half_spread"),
             py::arg("impact_k"), py::arg("scale"));

    m.def("run_backtest", [](const arr& R, const arr& R_all, long day0, py::array_t<long long, py::array::c_style | py::array::forcecast> reb_pos, const arr& S,
                             py::array_t<unsigned char, py::array::c_style | py::array::forcecast> OK, const arr& VOL, const arr& SIG, const arr& ADV,
                             arr REGM, double top, int weighting, bool long_short, double max_weight,
                             double capital, double cash_d, double vol_target, double min_trade,
                             const nrl::CostRates& rates) {
        auto [D, N] = shape2(R);
        const long K = static_cast<long>(reb_pos.size());
        std::vector<long> rp(K);
        for (long k = 0; k < K; ++k) rp[k] = static_cast<long>(reb_pos.at(k));
        nrl::BacktestOut o;
        {
            py::gil_scoped_release nogil;
            o = nrl::run_backtest(R.data(), D, R_all.data(), day0, N, rp.data(), K, S.data(), OK.data(), VOL.data(),
                                  SIG.data(), ADV.data(), REGM.data(), top, weighting, long_short, max_weight, capital,
                                  cash_d, vol_target, min_trade, rates);
        }
        py::array_t<double> W({K, N});
        std::copy(o.weights.begin(), o.weights.end(), W.mutable_data());
        py::dict d;
        d["nav"] = py::array_t<double>(o.nav.size(), o.nav.data());
        d["turnover"] = py::array_t<double>(K, o.turnover.data());
        d["cost"] = py::array_t<double>(K, o.cost.data());
        d["invested"] = py::array_t<double>(K, o.invested.data());
        d["value"] = py::array_t<double>(K, o.value.data());
        d["holdings"] = py::array_t<long>(K, o.holdings.data());
        d["traded"] = py::array_t<char>(K, o.traded.data());
        d["weights"] = W;
        return d;
    });

    py::class_<nrl::Quote>(m, "Quote")
        .def_readonly("price", &nrl::Quote::price).def_readonly("change_pct", &nrl::Quote::change_pct)
        .def_readonly("day_volume", &nrl::Quote::day_volume).def_readonly("ts_ms", &nrl::Quote::ts_ms)
        .def_readonly("version", &nrl::Quote::version);
    py::class_<nrl::Bar>(m, "Bar")
        .def_readonly("minute", &nrl::Bar::minute).def_readonly("open", &nrl::Bar::o).def_readonly("high", &nrl::Bar::h)
        .def_readonly("low", &nrl::Bar::l).def_readonly("close", &nrl::Bar::c).def_readonly("volume", &nrl::Bar::v);
    py::class_<nrl::TickStore>(m, "TickStore")
        .def(py::init<const std::vector<std::string>&, int>(), py::arg("symbols"), py::arg("bar_capacity") = 400)
        .def("update", &nrl::TickStore::update, py::arg("symbol"), py::arg("ts_ms"), py::arg("price"),
             py::arg("change_pct"), py::arg("day_volume"))
        .def("changed_since", &nrl::TickStore::changed_since)
        .def("quotes", &nrl::TickStore::quotes)
        .def("bars", &nrl::TickStore::bars)
        .def_property_readonly("ticks", &nrl::TickStore::ticks)
        .def_property_readonly("symbols", &nrl::TickStore::symbols)
        .def_property_readonly("memory_bytes", &nrl::TickStore::memory_bytes);
}
