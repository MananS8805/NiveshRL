// Live tick store for the desktop app: latest quote per symbol, change tracking,
// and 1-minute OHLCV bars aggregated from ticks, all in fixed memory.
//
// The WebSocket thread calls update() for every tick; the UI timer calls
// changed_since(version) to repaint only the symbols that moved. Memory is
// bounded: bars per symbol live in a ring of `bar_capacity` minutes, so the
// app can stream all day (and for days) without growing.
#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nrl {

struct Quote {
    double price = 0, change_pct = 0, day_volume = 0;
    int64_t ts_ms = 0;
    uint64_t version = 0;
};

struct Bar { int64_t minute = 0; double o = 0, h = 0, l = 0, c = 0, v = 0; };

class TickStore {
public:
    TickStore(const std::vector<std::string>& symbols, int bar_capacity = 400)
        : syms_(symbols), cap_(bar_capacity), quotes_(symbols.size()),
          bars_(symbols.size(), std::vector<Bar>(bar_capacity)), head_(symbols.size(), -1),
          count_(symbols.size(), 0), last_dayvol_(symbols.size(), -1) {
        for (size_t i = 0; i < syms_.size(); ++i) idx_[syms_[i]] = static_cast<int>(i);
    }

    // Returns false for an unknown symbol.
    bool update(const std::string& sym, int64_t ts_ms, double price, double change_pct, double day_volume) {
        auto it = idx_.find(sym);
        if (it == idx_.end()) return false;
        const int i = it->second;
        std::lock_guard<std::mutex> g(m_);
        Quote& q = quotes_[i];
        q.price = price; q.change_pct = change_pct; q.ts_ms = ts_ms; q.day_volume = day_volume;
        q.version = ++version_;
        ++ticks_;
        // 1-minute bars: volume is the increase in cumulative day volume.
        const int64_t minute = ts_ms / 60000;
        double dv = (last_dayvol_[i] >= 0 && day_volume >= last_dayvol_[i]) ? day_volume - last_dayvol_[i] : 0.0;
        last_dayvol_[i] = day_volume;
        auto& ring = bars_[i];
        if (head_[i] >= 0 && ring[head_[i]].minute == minute) {
            Bar& b = ring[head_[i]];
            b.h = std::max(b.h, price); b.l = std::min(b.l, price); b.c = price; b.v += dv;
        } else {
            head_[i] = (head_[i] + 1) % cap_;
            ring[head_[i]] = Bar{minute, price, price, price, price, dv};
            count_[i] = std::min(count_[i] + 1, cap_);
        }
        return true;
    }

    // Indices of symbols updated after `version`, and the current version.
    std::pair<std::vector<int>, uint64_t> changed_since(uint64_t version) const {
        std::lock_guard<std::mutex> g(m_);
        std::vector<int> out;
        for (size_t i = 0; i < quotes_.size(); ++i) if (quotes_[i].version > version) out.push_back(static_cast<int>(i));
        return {out, version_};
    }

    std::vector<Quote> quotes() const { std::lock_guard<std::mutex> g(m_); return quotes_; }

    // Oldest-first copy of a symbol's 1-minute bars.
    std::vector<Bar> bars(const std::string& sym) const {
        auto it = idx_.find(sym);
        if (it == idx_.end()) return {};
        const int i = it->second;
        std::lock_guard<std::mutex> g(m_);
        std::vector<Bar> out;
        const int n = count_[i];
        for (int k = n - 1; k >= 0; --k) out.push_back(bars_[i][((head_[i] - k) % cap_ + cap_) % cap_]);
        return out;
    }

    uint64_t ticks() const { return ticks_; }
    const std::vector<std::string>& symbols() const { return syms_; }
    size_t memory_bytes() const { return syms_.size() * (sizeof(Quote) + cap_ * sizeof(Bar)); }

private:
    std::vector<std::string> syms_;
    std::unordered_map<std::string, int> idx_;
    int cap_;
    std::vector<Quote> quotes_;
    std::vector<std::vector<Bar>> bars_;
    std::vector<int> head_, count_;
    std::vector<double> last_dayvol_;
    uint64_t version_ = 0, ticks_ = 0;
    mutable std::mutex m_;
};

}  // namespace nrl
