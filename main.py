"""
Entry point for surfalgo. Runs the test suite, tunes the urgency parameters,
estimates a worst-case lower bound, stress-tests 3 and 4 surfer heats, and
runs the full 10,000 heat simulation.

Usage:  python main.py
"""

import time

from surfalgo import (
    run_tests, optimize_urgency_params, estimate_competitive_ratio_bound,
    run_multi_surfer_simulation, run_simulation,
)


def main():
    print("Running tests")
    run_tests()

    print("\nTuning urgency parameters")
    best_params, _ = optimize_urgency_params(num_heats=2000)

    print("\nEstimating worst-case competitive ratio on adversarial streams")
    lower_bound, adv_avg = estimate_competitive_ratio_bound(num_trials=1000)
    print(f"5th percentile ratio: {lower_bound}")
    print(f"average on adversarial streams: {adv_avg}")

    print("\nMulti-surfer stress tests")
    for n in [3, 4]:
        r = run_multi_surfer_simulation(num_heats=1000, num_surfers=n)
        print(f"{n} surfers - win rates: {r['wins_by_surfer']}, "
              f"interference events: {r['interference_events']}, "
              f"avg ratio: {r['avg_competitive_ratio']}")

    print("\nRunning 10,000 heat simulation")
    t0 = time.time()
    results = run_simulation(num_heats=10000, urgency_params=best_params)
    total_time = round(time.time() - t0, 2)

    print()
    print(f"surfalgo win rate:         {results['win_rate']}%")
    print(f"Greedy win rate:          {results['greedy_win_rate']}%")
    print(f"surfalgo beats greedy:     {results['beats_greedy']}% of heats")
    print(f"Avg competitive ratio:    {results['avg_competitive_ratio']}")
    print(f"25th / median / 75th:     {results['p25_ratio']} / {results['p50_ratio']} / {results['p75_ratio']}")
    print(f"Avg surfalgo heat total:   {results['avg_ws_heat_total']}")
    print(f"Avg greedy heat total:    {results['avg_greedy_heat_total']}")
    print(f"Avg runtime per heat:     {results['avg_runtime_ms']} ms")
    print(f"Total simulation time:    {total_time}s")


if __name__ == "__main__":
    main()
