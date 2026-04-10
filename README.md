# Portfolio Optimization with Deep Reinforcement Learning

A Deep Q-Network (DQN) agent that learns to trade a portfolio of US stocks. This project reproduces the core method from Pigorsch & Schäfer (2021), *"High-Dimensional Stock Portfolio Trading with Deep Reinforcement Learning"*, and then extends it with three additions: a Sharpe-ratio reward bonus, a Hidden Markov Model for market-regime awareness, and Fama-French-style factor features.

**Authors:** Aryan Kalaskar, Manuel Torres, Crosby Sayan, Andres Cruz



## What the project does

Every day, for every stock in our universe, the agent looks at a snapshot of that stock (moving averages, volatility, fundamentals, a few Fama-French factor ranks, and the current market regime) and decides one thing: **hold cash, or be invested**. At the end of each day we equally weigh all the stocks the agent chose to hold, pay a small transaction cost on any new positions, and compound the returns. That gives us a daily portfolio return curve we can compare against buy-and-hold, momentum, and reversion benchmarks.



## Why Deep Reinforcement Learning

Traditional tabular Q-learning breaks down on real markets for two reasons: (1) the state space is continuous and high-dimensional, and (2) new stocks get issued all the time, so not every ticker has a full price history. DQN handles both problems naturally. It also mirrors how the Atari DQN work handled partial observability, which is a reasonable analogy for a trader who only sees a slice of the true market state.


## Project layout

```
portfolio-optimization-main/
├── configs/
│   └── default.yaml              Paper hyperparameters (reference values)
├── data/
│   ├── data_summary.ipynb        Exploratory data analysis of the raw dataset
│   └── stock_data.parquet        Raw equity data (not tracked in git)
├── notebooks/
│   └── feature_engineering.ipynb Prototype cleaning + normalization workflow
├── results/
│   ├── training_log.txt          Full stdout from the most recent training run
│   └── *.png                     Cumulative-return plots for each config
└── src/
    ├── main.py                   Entry point: trains and evaluates all configs
    ├── pipeline.py               Data loading, filtering, splits, z-scoring
    ├── environment.py            Custom Gym-style single-stock trading env
    ├── network.py                Q-network MLP and replay buffer
    ├── train.py                  DQN training loop with best-on-val checkpointing
    ├── evaluate.py               Backtesting, benchmarks, metrics, plots
    ├── factors.py                Fama-French-style factor ranks
    └── regime.py                 Gaussian HMM for market regime detection
```

---

## How it works (plain English)

### 1. Data pipeline (`pipeline.py`)

We start from a parquet file of US equity data covering 2010 through mid-2021 (~6.4M rows, 3,200+ stocks). The pipeline:

1. Drops rows with any missing features.
2. Keeps only stocks with at least 250 trading days of history.
3. Takes a **50-stock proof-of-concept subset**: the 20 largest by market cap, the 20 smallest, and 10 random mid-caps.
4. Splits chronologically: training < 2019, validation = 2019, test ≥ 2020.
5. Z-score normalizes features using **training-set statistics only** (no look-ahead).
6. Optionally adds factor ranks and regime labels.

**Features used as the agent's state (28 total, or 32 with factors):**

- *Technical (17):* SMA and EMA at 5/10/20/50/100/200 day windows, plus rolling standard deviations at 5/10/20/50/100 day windows.
- *Fundamental (10):* sales per share, operating margin, net profit margin, ROE, ROA, current ratio, debt ratio, book to market, quintile market cap, market cap.
- *Price (1):* close
- *Factor ranks (4, optional):* cross sectional percentile ranks for size, value, momentum, and quality.
- *Plus:* current position (cash or invested) and current market regime.

### 2. Trading environment (`environment.py`)

The environment follows the paper's per-asset MDP formulation. Each training episode focuses on one stock, chosen at random from the training universe, and steps forward day by day.

- **Action space:** `0 = cash`, `1 = invested`.
- **State:** the feature vector for that stock on that day, plus current position and current regime.
- **Reward when invested:** the stock's next day return, minus a transaction cost if this is a new position.
- **Reward when in cash:** the cross sectional mean return across all stocks that day. This is the paper's trick for making "doing nothing" a meaningful signal, the agent is punished for sitting out a rally.

We implemented three reward variants:

| Variant | What it adds |
|---|---|
| `base` | Pure paper reward (stock return - transaction cost). |
| `sharpe` | Adds a bonus proportional to the stock's rolling 20 day Sharpe ratio. |
| `sharpe+regime` | Same as above, but the Sharpe bonus is scaled by a regime weight so the agent is rewarded more for risk taking in bear regimes. |

### 3. Market regime model (`regime.py`)

We fit a 3-state Gaussian HMM from scratch in PyTorch (forward-backward, EM, and Viterbi are all hand-rolled — no `hmmlearn` dependency). The observations are daily cross-sectional mean return and cross-sectional volatility.

The HMM learns three distinct regimes that roughly correspond to *bear* (negative mean, moderate volatility), *high-volatility* (positive mean, very high vol), and *calm* (slightly positive mean, low vol). Each date in every split gets both a hard regime label and soft regime probabilities, which the environment uses to scale the Sharpe bonus.

### 4. Factor model (`factors.py`)

Four cross-sectional percentile ranks, recomputed per trading day:

- **Size:** rank on market cap (SMB proxy).
- **Value:** rank on book-to-market (HML proxy).
- **Momentum:** rank on the ratio of 5-day to 200-day SMA (UMD proxy).
- **Quality:** rank on the average of ROE and ROA (RMW proxy).

A rank of 1.0 means the stock is at the top of the cross-section on that day; 0.0 means the bottom. These are deliberately simple compared to commercial factor models like Barra or Axioma.

### 5. DQN agent (`network.py`, `train.py`)

- **Architecture:** a 2-hidden-layer ReLU MLP with output dimension 2 (one Q-value per action). Hidden width is configurable (we use 32, 64, and 128).
- **Target network:** synced every 1,000 steps to stabilize learning.
- **Replay buffer:** 50,000 transitions (scaled down from the paper's 300,000).
- **Optimizer:** Adam, learning rate 5e-4, gradient clipping with max norm 1.0.
- **Training loop:** 500,000 environment steps per agent (the paper uses 3M), epsilon-greedy with epsilon 0.3, one gradient update every 20 environment steps, batch size 1024.
- **Validation checkpointing:** every 10,000 steps we run the current network on the validation year and keep whichever weights produced the best cumulative return. This matches Algorithm 1, lines 8-14 in the paper.
- **Ensembling:** we train three independent networks (widths 32, 64, 128) per configuration, and at test time we use majority voting across the three.

### 6. Evaluation and benchmarks (`evaluate.py`)

On the test set, for each trading day:

1. Each of the three ensemble networks votes on every stock.
2. A stock is included in the portfolio if at least two of the three networks vote to invest.
3. We equal-weight the selected stocks, pay transaction costs on new positions, and record the portfolio return.
4. If the agent picks fewer than three stocks on a given day, we fall back to holding the full buy-and-hold basket to avoid unrealistic concentration.

We compare against three benchmarks from the paper: buy-and-hold, a 5-day momentum rule, and a 5-day reversion rule. Metrics reported: cumulative return, annualized Sharpe, maximum drawdown, and win rate against each benchmark.

---

## Results

All configurations were trained and tested on the same 50-stock subset. The test period starts January 2020, which includes the COVID crash and recovery.

| Configuration | Cumulative Return | Sharpe | Max Drawdown | Win Rate vs B&H |
|---|---:|---:|---:|---:|
| Buy & Hold | +114.52% | 1.65 | -31.62% | — |
| 5-day Momentum | -21.41% | -0.21 | -62.46% | — |
| 5-day Reversion | +141.01% | 1.60 | -29.51% | — |
| DQN, no factors, base reward | +189.49% | 1.52 | -28.81% | 44.6% |
| **DQN, no factors, sharpe reward** | **+303.28%** | **2.27** | **-21.44%** | **52.8%** |
| DQN, no factors, sharpe + regime | +296.44% | 1.53 | -37.06% | 55.7% |
| DQN, with factors, sharpe + regime | +225.48% | 1.87 | -21.44% | 48.8% |

The Sharpe-reward agent is the standout by risk-adjusted return: roughly 2.7x buy-and-hold with a shallower drawdown. Adding regime awareness and factors produced mixed results on this particular test window, which is something we want to investigate further.

See `results/*.png` for equity curves.

---

## How to run it

### Requirements

- Python 3.10+
- PyTorch
- pandas, numpy, pyarrow
- matplotlib

### Data

The pipeline expects a parquet file at `data/stock_data.parquet` with the following columns per `(ticker, date)`: `ret`, `close`, all the technical and fundamental columns listed in `src/pipeline.py`. The raw data itself is not tracked in git.

### Train and evaluate

From the `src/` directory:

```bash
python main.py
```

This will sequentially train all four configurations, save model checkpoints to `results/qnet_*.pt`, and produce equity-curve plots in `results/`. Expect several hours on a laptop CPU; a single GPU brings it down significantly.

---

## Project milestones

1. **Data pipeline, environment, and backtesting.** Acquire and process US equity data, build the custom Gym environment, stand up the backtesting harness. ✅
2. **Core algorithm.** Implement the DQN architecture, training loop, and ensemble evaluation following the paper. ✅
3. **Extensions.** Add risk-aware rewards, HMM regime sensitivity, and a lightweight factor model. ✅
4. **Analysis and write-up.** Investigate why regime and factor additions underperformed the pure Sharpe reward on this test window; stress-test on out-of-sample windows including 2008.

---

## References

- Pigorsch, U. and Schäfer, S. (2021). *High-Dimensional Stock Portfolio Trading with Deep Reinforcement Learning.*
- Hambly, B., Xu, R., and Yang, H. (2023). *Recent Advances in Reinforcement Learning in Finance.*
- Choudhary, V. et al. (2025). Risk-adjusted reward functions for DRL portfolio optimization.
- Nixon, J. (2025). Hidden Markov Models for market regime detection.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* (Original DQN paper.)
