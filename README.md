# Surfalgo

In WSL competitive surfing, two surfers share a 30-minute heat and compete for unpredictable waves, where only the best two scores out of 20.0 count. The surfer with priority must decide in real time whether to take an incoming wave or pass to preserve priority and block their opponent this is all without knowing what waves come next. My algorithm formalizes this take or pass decision as a scheduling problem, helping a surfer maximize their chance of winning the heat under uncertainty. It models official WSL rules including priority rotation, two wave scoring, and interference penalties.

## How it works
The algorithm has three parts:

A min heap tracks surfers by priority and rotates priority
  in O(log N) after every wave taken.
An online decision engine decides take or pass based on score gap, time
  remaining, and a time decaying urgency threshold.
A hindsight solver sees all waves in advance provides an upper bound to
  measure how close the online algorithm gets to optimal the competitive ratio.



## Build and run
Run the full pipeline

```bash
python main.py
```

or:

```bash
make run
```

## Tests

```bash
make test
```

This runs the automated suite in test_surfalgo.py 12 tests covering the
decision rules, scoring, interference, priority rotation, and calibration. It uses pytest if available otherwise a built in runner.
You can also run it directly:

```bash
python test_surfalgo.py
```

## Files
surfalgo.py - core algorithm and simulation functions
main.py - entry point that runs the full pipeline
test_surfalgo.py - automated test suite
Makefile - 'make test' and 'make run' shortcuts

## Reproducing results
All wave streams are generated from fixed random seeds, so results are
deterministic across runs. 'python main.py' reproduces the figures reported in
the write-up. 

## Results
Across 10,000 simulated heats my algo reaches about 92% of the hindsight
solver performance, beats a greedy baseline in
roughly 37% of shared wave streams, and processes a full heat in well under a
millisecond.
