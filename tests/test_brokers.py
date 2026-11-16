import pytest

from niveshrl import brokers, paper


def test_default_is_paper_and_fills(tmp_path):
    b = brokers.PaperBroker(path=tmp_path / "p.json")
    assert isinstance(b, brokers.Broker) and not b.is_live
    b.place(paper.Order("TCS.NS", "BUY", 2))
    events = b.sync({"TCS.NS": 100.0})
    assert events and b.positions()["TCS.NS"].qty == 2
    assert paper.load(tmp_path / "p.json").positions["TCS.NS"].qty == 2      # persisted


@pytest.mark.parametrize("name", sorted(brokers.SLOTS))
def test_live_slots_refuse_orders(name):
    b = brokers.get_broker({"provider": name})
    assert b.is_live and b.status()["connected"] is False
    with pytest.raises(brokers.LiveTradingDisabled):
        b.place(paper.Order("TCS.NS", "BUY", 1))
    with pytest.raises(brokers.BrokerNotConfigured):
        b.positions()


def test_unknown_provider_falls_back_to_paper():
    assert brokers.get_broker({"provider": "made-up"}).name == "paper"
