import numpy as np
import advisory as A

THR = 0.5


def tier(p, now=30):
    return A.tier_array([p], [now], THR)[0]


def test_tiers_by_probability():
    assert tier(0.10) == "LOW_RISK"
    assert tier(0.49) == "LOW_RISK"
    assert tier(0.50) == "WARNING"
    assert tier(0.99) == "WARNING"


def test_current_high_reading_overrides():
    assert tier(0.01, now=120) == "ELEVATED_NOW"
    assert tier(0.99, now=91) == "ELEVATED_NOW"
    assert tier(0.99, now=90) == "WARNING"     # exactly 90 is not above the threshold


def test_advisory_never_negative_and_has_caveats():
    a = A.make_advisory(0.6, 50, 70, -10, 120, THR)
    assert a.tier == "WARNING"
    assert a.typical_range[0] == 0
    assert any("not followed by a spike" in c for c in a.caveats)
    assert any("not an official" in c.lower() for c in a.caveats)


def test_low_risk_does_not_promise_safety():
    a = A.make_advisory(0.05, 20, 25, 5, 60, THR)
    assert a.tier == "LOW_RISK"
    assert any("does not guarantee" in c for c in a.caveats)


def test_every_tier_has_text():
    for t in A.TIERS:
        assert A.HEADLINES[t] and A.ACTIONS[t]