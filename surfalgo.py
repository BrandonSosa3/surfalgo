"""
My surfalgo is an online scheduling algorithm for competitive surf heats.
Import this module and call the functions, or run main.py for the full pipeline.
"""

import heapq
import time
import random
from itertools import combinations


WAVE_WEIGHTS = {
    'point_break': (0.7, 0.1, 0.2),
    'barrel':      (0.1, 0.7, 0.2),
    'beach_break': (0.3, 0.2, 0.5),
}


class Surfer:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority
        self.scores = []
        self.interference = False

    def heat_total(self):
        top2 = sorted(self.scores, reverse=True)[:2]
        if self.interference and len(top2) == 2:
            return top2[0]
        return sum(top2)

    def worst_counting_score(self):
        top2 = sorted(self.scores, reverse=True)[:2]
        return min(top2) if len(top2) == 2 else 0

    def needs_score(self):
        return len(self.scores) < 2

    def __lt__(self, other):
        return self.priority < other.priority


def score_wave(wave_type, quality, weights=None):
    table = weights if weights else WAVE_WEIGHTS
    w = table.get(wave_type, (0.33, 0.33, 0.34))
    return round(quality * sum(w), 2)


def calibrate_weights(historical_data):
    by_type = {}
    for wave_type, quality, actual in historical_data:
        by_type.setdefault(wave_type, []).append((quality, actual))

    calibrated = {}
    for wave_type, points in by_type.items():
        num = sum(q * s for q, s in points)
        den = sum(q * q for q, s in points)
        scale = num / den if den > 0 else 1.0
        prior = WAVE_WEIGHTS.get(wave_type, (0.33, 0.33, 0.34))
        total = sum(prior)
        calibrated[wave_type] = tuple(round((p / total) * scale, 4) for p in prior)

    for wt in WAVE_WEIGHTS:
        calibrated.setdefault(wt, WAVE_WEIGHTS[wt])
    return calibrated


def urgency_factor(time_remaining, heat_duration=1800,
                   early_threshold=0.5, mid_threshold=0.2,
                   early_val=1.0, mid_val=0.75, late_val=0.5):
    t = time_remaining / heat_duration
    if t > early_threshold:
        return early_val
    elif t > mid_threshold:
        return mid_val
    return late_val


def should_take(wave_potential, surfer, opponent,
                time_remaining, num_opponents=1, urgency_params=None):
    params = urgency_params or {}
    opp_total = opponent.heat_total()
    self_total = surfer.heat_total()
    urgency = urgency_factor(time_remaining, **params)

    if time_remaining < 180 and surfer.needs_score():
        return True

    if self_total > opp_total and not surfer.needs_score():
        return wave_potential > surfer.worst_counting_score()

    scale = 1.0 + (num_opponents - 1) * 0.05

    if len(surfer.scores) == 0:
        return wave_potential >= (5.0 * urgency * scale)

    projected = sum(sorted(surfer.scores + [wave_potential], reverse=True)[:2])
    threshold = (opp_total + 0.01) * urgency * scale
    return projected >= threshold


def greedy_should_take(wave_potential, surfer, opponent, time_remaining, **kwargs):
    return True


def hindsight_solver(opponent_total, all_waves, weights=None):
    scored = [score_wave(wt, q, weights) for wt, q, _ in all_waves]
    n = len(scored)
    best_total = 0
    best_margin = -999

    for r in [2, 1]:
        for combo in combinations(range(n), r):
            total = sum(scored[i] for i in combo)
            margin = total - opponent_total
            if total > opponent_total and margin > best_margin:
                best_margin = margin
                best_total = total
            elif best_margin < 0 and total > best_total:
                best_total = total
    return round(best_total, 2)


def run_heat(surfers, waves, strategy_fn, weights=None):
    heap = list(surfers)
    heapq.heapify(heap)
    decisions = []
    num_surfers = len(surfers)

    for wave_type, quality, t_remaining in waves:
        if len(heap) < 2:
            break

        wave_pot = score_wave(wave_type, quality, weights)
        top = heap[0]
        rivals = [s for s in heap if s.name != top.name]
        opponent = max(rivals, key=lambda s: s.heat_total()) if rivals else None
        if opponent is None:
            break

        take = strategy_fn(wave_pot, top, opponent, t_remaining,
                           num_opponents=num_surfers - 1)

        if take:
            heapq.heappop(heap)
            top.scores.append(wave_pot)
            top.priority = max(s.priority for s in heap) + 1
            heapq.heappush(heap, top)
            decisions.append((top.name, wave_pot, 'TOOK'))
        else:
            decisions.append((top.name, wave_pot, 'PASSED'))

    return decisions


def generate_wave_stream(num_waves=20, heat_duration=1800, seed=None):
    rng = random.Random(seed)
    wave_types = ['point_break', 'barrel', 'beach_break']
    waves = []
    for i in range(num_waves):
        t_remaining = heat_duration - (i * heat_duration / num_waves)
        wave_type = rng.choice(wave_types)
        quality = min(10.0, max(1.0, rng.gauss(5.5, 1.8)))
        waves.append((wave_type, round(quality, 2), round(t_remaining)))
    return waves


def optimize_urgency_params(num_heats=2000, num_waves=20, verbose=True):
    early_vals = [0.9, 1.0, 1.1]
    mid_vals = [0.6, 0.75, 0.85]
    late_vals = [0.4, 0.5, 0.6]
    best_ratio = 0
    best_params = {}

    if verbose:
        print(f"Searching urgency parameters over {num_heats} heats each...")

    for ev in early_vals:
        for mv in mid_vals:
            for lv in late_vals:
                if not (ev >= mv >= lv):
                    continue
                ratios = []
                for i in range(num_heats):
                    waves = generate_wave_stream(num_waves, seed=i)
                    ws = Surfer("WS", 0)
                    opp = Surfer("Opp", 1)

                    def make_strategy(e, m, l):
                        def strategy(wave_pot, surfer, opponent, t_rem, **kwargs):
                            return should_take(
                                wave_pot, surfer, opponent, t_rem,
                                urgency_params={'early_val': e, 'mid_val': m, 'late_val': l})
                        return strategy

                    run_heat([ws, opp], waves, make_strategy(ev, mv, lv))
                    oracle = hindsight_solver(opp.heat_total(), waves)
                    if oracle > 0:
                        ratios.append(min(1.0, ws.heat_total() / oracle))

                avg = sum(ratios) / len(ratios) if ratios else 0
                if avg > best_ratio:
                    best_ratio = avg
                    best_params = {'early_val': ev, 'mid_val': mv, 'late_val': lv}

    if verbose:
        print(f"Best competitive ratio: {round(best_ratio, 4)}")
        print(f"Best params: {best_params}")
    return best_params, round(best_ratio, 4)


def estimate_competitive_ratio_bound(num_trials=1000, num_waves=20):
    ratios = []
    for trial in range(num_trials):
        rng = random.Random(trial + 99999)
        waves = []
        for i in range(num_waves):
            t_remaining = 1800 - (i * 1800 / num_waves)
            wave_type = rng.choice(['point_break', 'barrel', 'beach_break'])
            if i < num_waves * 0.6:
                quality = rng.uniform(3.5, 6.0)
            else:
                quality = rng.uniform(7.5, 10.0)
            waves.append((wave_type, round(quality, 2), round(t_remaining)))

        ws = Surfer("WS", 0)
        opp = Surfer("Opp", 1)
        run_heat([ws, opp], waves, should_take)
        hs = hindsight_solver(opp.heat_total(), waves)
        if hs > 0:
            ratios.append(min(1.0, ws.heat_total() / hs))

    ratios.sort()
    idx = int(len(ratios) * 0.05)
    return round(ratios[idx], 4), round(sum(ratios) / len(ratios), 4)


def run_multi_surfer_simulation(num_heats=1000, num_surfers=3, num_waves=20, verbose=True):
    wins = {f"Surfer_{i}": 0 for i in range(num_surfers)}
    interference_events = 0
    ratios = []

    for heat_i in range(num_heats):
        waves = generate_wave_stream(num_waves, seed=heat_i + 50000)
        surfers = [Surfer(f"Surfer_{i}", i) for i in range(num_surfers)]
        run_heat(surfers, waves, should_take)

        rng = random.Random(heat_i)
        if rng.random() < 0.05:
            rng.choice(surfers).interference = True
            interference_events += 1

        winner = max(surfers, key=lambda s: s.heat_total())
        wins[winner.name] += 1

        best_opp = max(s.heat_total() for s in surfers if s.name != "Surfer_0")
        hs = hindsight_solver(best_opp, waves)
        if hs > 0:
            ratios.append(min(1.0, surfers[0].heat_total() / hs))

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    return {
        "wins_by_surfer": {k: round(v / num_heats * 100, 1) for k, v in wins.items()},
        "interference_events": interference_events,
        "avg_competitive_ratio": round(avg_ratio, 4),
    }


def run_simulation(num_heats=10000, num_waves=20, urgency_params=None, weights=None, verbose=True):
    ws_wins = greedy_wins = ws_beats_greedy = 0
    ratios = []
    ws_totals = []
    gr_totals = []
    runtimes = []
    params = urgency_params or {}

    def ws_strategy(wave_pot, surfer, opponent, t_rem, **kwargs):
        return should_take(wave_pot, surfer, opponent, t_rem,
                           urgency_params=params, **kwargs)

    for i in range(num_heats):
        waves = generate_wave_stream(num_waves, seed=i)

        ws = Surfer("WS", 0)
        opp = Surfer("Opp", 1)
        t0 = time.time()
        run_heat([ws, opp], waves, ws_strategy, weights)
        runtimes.append(time.time() - t0)
        ws_total = ws.heat_total()
        opp_total = opp.heat_total()
        ws_totals.append(ws_total)

        gr = Surfer("GR", 0)
        gr_opp = Surfer("Opp", 1)
        run_heat([gr, gr_opp], waves, greedy_should_take, weights)
        gr_total = gr.heat_total()
        gr_totals.append(gr_total)

        hs_total = hindsight_solver(opp_total, waves, weights)

        if ws_total > opp_total:
            ws_wins += 1
        if gr_total > gr_opp.heat_total():
            greedy_wins += 1
        if ws_total > gr_total:
            ws_beats_greedy += 1
        if hs_total > 0:
            ratios.append(min(1.0, ws_total / hs_total))

        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i+1}/{num_heats} heats done")

    sorted_ratios = sorted(ratios)
    return {
        "num_heats": num_heats,
        "win_rate": round(ws_wins / num_heats * 100, 2),
        "greedy_win_rate": round(greedy_wins / num_heats * 100, 2),
        "beats_greedy": round(ws_beats_greedy / num_heats * 100, 2),
        "avg_competitive_ratio": round(sum(ratios) / len(ratios), 4),
        "p25_ratio": round(sorted_ratios[int(len(sorted_ratios) * 0.25)], 4),
        "p50_ratio": round(sorted_ratios[int(len(sorted_ratios) * 0.50)], 4),
        "p75_ratio": round(sorted_ratios[int(len(sorted_ratios) * 0.75)], 4),
        "avg_ws_heat_total": round(sum(ws_totals) / len(ws_totals), 2),
        "avg_greedy_heat_total": round(sum(gr_totals) / len(gr_totals), 2),
        "avg_runtime_ms": round(sum(runtimes) / len(runtimes) * 1000, 4),
    }


def run_tests():
    passed = 0
    failed = 0

    def check(label, got, expected):
        nonlocal passed, failed
        if str(got) == str(expected):
            passed += 1
            print(f"ok   - {label}")
        else:
            failed += 1
            print(f"FAIL - {label} (expected {expected}, got {got})")

    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [7.5, 7.0]; b.scores = [6.0]
    d = run_heat([b, a], [("beach_break", 3.0, 900)], should_take)
    check("strategic pass on a weak wave", d[0][2], "PASSED")

    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [7.5, 7.0]; b.scores = [6.0]
    d = run_heat([b, a], [("beach_break", 5.5, 150)], should_take)
    check("desperation mode under 3 minutes", d[0][2], "TOOK")

    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [6.0, 5.5]; b.scores = [8.5, 7.5]
    d = run_heat([b, a], [("point_break", 6.0, 900)], should_take)
    check("leading, weak wave does not improve total", d[0][2], "PASSED")

    a = Surfer("A", 1); b = Surfer("B", 0)
    a.scores = [6.0, 5.5]; b.scores = [8.5, 7.5]
    d = run_heat([b, a], [("barrel", 9.2, 900)], should_take)
    check("leading, strong wave improves total", d[0][2], "TOOK")

    b = Surfer("B", 0); b.scores = [8.5, 7.0]; b.interference = True
    check("interference penalty uses best wave only", b.heat_total(), 8.5)
    b2 = Surfer("B2", 0); b2.scores = [8.5, 7.0]
    check("normal total sums best two", b2.heat_total(), 15.5)

    for wt in ['point_break', 'barrel', 'beach_break']:
        check(f"{wt} quality 8.0 scores 8.0", score_wave(wt, 8.0), 8.0)

    a = Surfer("A", 1); b = Surfer("B", 0)
    waves = [("point_break", 9.5, 1200), ("barrel", 8.0, 1170)]
    d = run_heat([b, a], waves, should_take)
    check("priority rotation: B acts first", d[0][0], "B")
    check("priority rotation: A acts second", d[1][0], "A")

    waves = generate_wave_stream(15, seed=42)
    ws = Surfer("WS", 0); opp = Surfer("Opp", 1)
    run_heat([ws, opp], waves, should_take)
    hs = hindsight_solver(opp.heat_total(), waves)
    ratio = min(1.0, ws.heat_total() / hs) if hs > 0 else 1.0
    check("competitive ratio above 0.8 on seed 42", ratio > 0.8, True)

    historical = [
        ("barrel", 8.0, 7.5), ("barrel", 6.0, 5.8),
        ("point_break", 7.0, 6.9), ("point_break", 9.0, 8.8),
        ("beach_break", 5.0, 4.9), ("beach_break", 8.5, 8.2),
    ]
    cal = calibrate_weights(historical)
    check("calibrated barrel scale below 1.0", sum(cal["barrel"]) < 1.0, True)

    surfers = [Surfer(f"S{i}", i) for i in range(3)]
    waves = generate_wave_stream(10, seed=7)
    d = run_heat(surfers, waves, should_take)
    check("3-surfer heat decisions match wave count", len(d), 10)

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0
