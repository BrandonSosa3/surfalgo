"""
Test for surfalgo.
"""

from surfalgo import (
    Surfer, score_wave, calibrate_weights, urgency_factor,
    should_take, hindsight_solver, run_heat, generate_wave_stream,
)


def test_strategic_pass_weak_wave():
    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [7.5, 7.0]; b.scores = [6.0]
    d = run_heat([b, a], [("beach_break", 3.0, 900)], should_take)
    assert d[0][2] == "PASSED"


def test_desperation_mode():
    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [7.5, 7.0]; b.scores = [6.0]
    d = run_heat([b, a], [("beach_break", 5.5, 150)], should_take)
    assert d[0][2] == "TOOK"


def test_leading_no_upgrade():
    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [6.0, 5.5]; b.scores = [8.5, 7.5]
    d = run_heat([b, a], [("point_break", 6.0, 900)], should_take)
    assert d[0][2] == "PASSED"


def test_leading_with_upgrade():
    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [6.0, 5.5]; b.scores = [8.5, 7.5]
    d = run_heat([b, a], [("barrel", 9.2, 900)], should_take)
    assert d[0][2] == "TOOK"


def test_interference_penalty():
    b = Surfer("B", 0); b.scores = [8.5, 7.0]; b.interference = True
    assert b.heat_total() == 8.5


def test_normal_total():
    b = Surfer("B", 0); b.scores = [8.5, 7.0]
    assert b.heat_total() == 15.5


def test_wave_scoring_scale():
    for wt in ['point_break', 'barrel', 'beach_break']:
        assert score_wave(wt, 8.0) == 8.0


def test_priority_rotation():
    a = Surfer("A", 1); b = Surfer("B", 0)
    waves = [("point_break", 9.5, 1200), ("barrel", 8.0, 1170)]
    d = run_heat([b, a], waves, should_take)
    assert d[0][0] == "B"
    assert d[1][0] == "A"


def test_competitive_ratio_reasonable():
    waves = generate_wave_stream(15, seed=42)
    ws = Surfer("WS", 0); opp = Surfer("Opp", 1)
    run_heat([ws, opp], waves, should_take)
    hs = hindsight_solver(opp.heat_total(), waves)
    ratio = min(1.0, ws.heat_total() / hs) if hs > 0 else 1.0
    assert ratio > 0.8


def test_weight_calibration():
    historical = [
        ("barrel", 8.0, 7.5), ("barrel", 6.0, 5.8),
        ("point_break", 7.0, 6.9), ("point_break", 9.0, 8.8),
        ("beach_break", 5.0, 4.9), ("beach_break", 8.5, 8.2),
    ]
    cal = calibrate_weights(historical)
    assert "barrel" in cal and "point_break" in cal
    assert sum(cal["barrel"]) < 1.0


def test_three_surfer_heat():
    surfers = [Surfer(f"S{i}", i) for i in range(3)]
    waves = generate_wave_stream(10, seed=7)
    d = run_heat(surfers, waves, should_take)
    assert len(d) == 10


def test_urgency_decay():
    assert urgency_factor(1800) >= urgency_factor(900) >= urgency_factor(60)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"ok   - {t.__name__}")
            passed += 1
        except AssertionError:
            print(f"FAIL - {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
