# bandit

Fast implementation of a **spatial combinatorial multi-armed bandit** (CMAB): each round an agent picks a subset of `k` arms, receives the sum of their rewards and pays a spatial penalty that grows with the distance between the chosen arms.

## Contents

| File | Description |
| --- | --- |
| `allinone.py` | Complete experiment: `SpatialMABEnvironment`, `UCBSubsetStrategy` (UCB with an optional decaying epsilon-greedy factor), vectorised reward computation, subset cache with Monte-Carlo fallback for large problems, training loop and reward plots. `gamma = 0` gives pure UCB; `gamma > 0` gives a UCB / epsilon-greedy hybrid. |
| `super_optimum_cmab.py` | Compact variant with an `Agent` class and an `argparse` command line interface (`--steps`, `--verbose`, `--monitor`). |

## Requirements

Python 3.9+, `numpy`, `matplotlib`.

```bash
pip install numpy matplotlib
python allinone.py
python super_optimum_cmab.py --help
```

## Related work

[`RL-sim`](https://github.com/XioDr/RL-sim) contains agents with different levels of knowledge about the environment.
