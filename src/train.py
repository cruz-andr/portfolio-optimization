"""
PoC v2: Paper-faithful Deep Q-Learning for Portfolio Trading
Matches Pigorsch & Schäfer (2021) as closely as possible at small scale.

Paper spec:
  - Features: 12 MAs (SMA+EMA, windows 5/10/20/50/100/200) of returns,
    5 rolling stds of returns, 10 fundamentals, close price, position dummy
  - Hyperparams: gamma=0.9, epsilon=0.3, 3M steps, replay=300k, batch=1024,
    grad every 20, eval every 10k, Adam, ensemble of 32/64/128
  - Reward: asset return for invest, cross-sectional mean for cash (eq. 3)
  - Validation: save best model by cumulative return on val set

Our scale-down:
  - 50 stocks instead of 500
  - 500k steps instead of 3M (per agent)
  - Same everything else
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
from environment import TransactionEnvironment
from regime import prepare_regimes
from factors import add_factor_ranks, FACTOR_COLS


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=300_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        dim = len(s[0])
        ns = [n if n is not None else np.zeros(dim) for n in ns]
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


TECHNICAL_COLS = [
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'rstd_5', 'rstd_10', 'rstd_20', 'rstd_50', 'rstd_100'
]

FUNDAMENTAL_COLS = [
    'sales_per_share', 'operating_margin', 'net_profit_margin',
    'roe', 'roa', 'current_ratio', 'debt_ratio',
    'book_to_market', 'mkt_cap_q', 'mkt_cap'
]

PRICE_COLS = ['close']

FEATURE_COLS = TECHNICAL_COLS + FUNDAMENTAL_COLS + PRICE_COLS + FACTOR_COLS


def prepare_data(parquet_path='../data/stock_data.parquet'):
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    before = len(df)
    BASE_FEATURE_COLS = TECHNICAL_COLS + FUNDAMENTAL_COLS + PRICE_COLS
    df = df.dropna(subset=BASE_FEATURE_COLS)
    print(f"Dropped {before - len(df)} NaN rows ({len(df)} remaining)")

    counts = df.groupby('ticker').size()
    valid = counts[counts >= 250].index
    df = df[df['ticker'].isin(valid)]
    print(f"Stocks with 250+ days: {len(valid)}")

    # PoC subset: 50 stocks (20 big, 20 small, 10 random)
    latest = df.sort_values('date').groupby('ticker').tail(1)
    by_cap = latest.sort_values('mkt_cap', ascending=False)
    big = by_cap.head(20)['ticker'].tolist()
    small = by_cap.tail(20)['ticker'].tolist()
    mid = by_cap[~by_cap['ticker'].isin(big + small)]
    rand = mid.sample(min(10, len(mid)), random_state=42)['ticker'].tolist()
    poc_tickers = list(set(big + small + rand))
    df = df[df['ticker'].isin(poc_tickers)]

    print(f"PoC stocks: {len(poc_tickers)}")
    print(f"Total rows: {len(df)}")

    # paper split: train < 2019, val 2019, test 2020+
    df_train = df[df['date'] < '2019-01-01'].copy()
    df_val = df[(df['date'] >= '2019-01-01') & (df['date'] < '2020-01-01')].copy()
    df_test = df[df['date'] >= '2020-01-01'].copy()
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # paper: z-score normalize using training set statistics
    BASE_FEATURE_COLS = TECHNICAL_COLS + FUNDAMENTAL_COLS + PRICE_COLS
    train_mean = df_train[BASE_FEATURE_COLS].mean()
    train_std = df_train[BASE_FEATURE_COLS].std().replace(0, 1)
    for split in [df_train, df_val, df_test]:
        split[BASE_FEATURE_COLS] = (split[BASE_FEATURE_COLS] - train_mean) / train_std


    df_train = add_factor_ranks(df_train)
    df_val   = add_factor_ranks(df_val)
    df_test  = add_factor_ranks(df_test)

    # paper eq. 3: cash reward = cross-sectional mean return
    for split in [df_train, df_val, df_test]:
        split['cross_sect_mean_ret'] = split.groupby('date')['ret'].transform('mean')
        
    _, df_train, df_val, df_test, regime_weights = prepare_regimes(df_train, df_val, df_test)

    print(f"State dim: {len(FEATURE_COLS)} features + 1 dummy = {len(FEATURE_COLS)+1}")
    return df_train, df_val, df_test, FEATURE_COLS, regime_weights


def train_agent(
    df_train,
    df_val,
    feature_cols,
    transaction_cost=0.0005,
    n_steps=500_000,
    gamma=0.9,
    epsilon=0.3,
    batch_size=1024,
    grad_interval=20,
    eval_interval=10_000,
    target_update_interval=1000,
    hidden_dim=64,
    reward_fn="base",
    sharpe_lambda=0.1,
    regime_weights=None
):
    env = TransactionEnvironment(df_train, feature_cols, transaction_cost, reward_fn=reward_fn, sharpe_lambda=sharpe_lambda, regime_weights=regime_weights)
    val_env = TransactionEnvironment(df_val, feature_cols, transaction_cost, reward_fn=reward_fn, sharpe_lambda=sharpe_lambda, regime_weights=regime_weights)
    state_dim = len(feature_cols) + 2 
    q_net = QNetwork(state_dim, hidden_dim)
    target_net = QNetwork(state_dim, hidden_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=5e-4)
    buffer = ReplayBuffer(capacity=300_000)

    best_val_cr = 0.0
    best_params = None
    train_rewards = []

    state = env.reset()
    ep_reward = 0.0

    for step in range(1, n_steps + 1):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = q_net(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        ep_reward += reward

        if done:
            train_rewards.append(ep_reward)
            ep_reward = 0.0
            state = env.reset()
        else:
            state = next_state

        # gradient step every 20 iterations
        if step % grad_interval == 0 and len(buffer) >= batch_size:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)
            q_vals = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(ns_b).max(1)[0]
                targets = r_b + gamma * next_q * (1 - d_b)
            loss = nn.MSELoss()(q_vals, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

        if step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        # validation checkpoint (paper Algorithm 1, lines 8-14)
        if step % eval_interval == 0:
            val_cr = evaluate_cumulative_return(val_env, q_net, feature_cols)
            marker = ""
            if val_cr > best_val_cr:
                best_val_cr = val_cr
                best_params = {k: v.clone() for k, v in q_net.state_dict().items()}
                marker = " ::::: new best"
            print(f"Step {step:>7d} | Val CR: {val_cr:+.2%} | Best: {best_val_cr:+.2%}{marker}")

    if best_params is not None:
        q_net.load_state_dict(best_params)
        print(f"\nRestored best model (Val CR: {best_val_cr:+.2%})")
    else:
        print("\nNo positive validation CR found, using final weights")

    return q_net, train_rewards


def evaluate_cumulative_return(env, q_net, feature_cols):
    """Paper-style eval: daily equal-weight portfolio, compounded."""
    df = env.stockData
    dates = sorted(df['date'].unique())
    prev_inv = set()
    daily_rets = []

    for date in dates:
        dd = df[df['date'] == date]
        if len(dd) < 2:
            continue
        inv = []
        for _, row in dd.iterrows():
            feat = row[feature_cols].values.astype(float)
            pos = 1 if row['ticker'] in prev_inv else 0
            regime = row['regime']
            st = np.append(feat, [pos, regime])
            with torch.no_grad():
                a = q_net(torch.FloatTensor(st).unsqueeze(0)).argmax(1).item()
            if a == 1:
                inv.append(row['ticker'])

        if inv:
            r = dd[dd['ticker'].isin(inv)]['ret']
            new = set(inv) - prev_inv
            cost = len(new) / len(inv) * env.transactionCost
            daily_rets.append(r.mean() - cost)
        else:
            daily_rets.append(dd['cross_sect_mean_ret'].iloc[0])
        prev_inv = set(inv)

    return np.prod([1 + r for r in daily_rets]) - 1


def evaluate_portfolio(df_test, q_nets, feature_cols, tc=0.0005, min_picks=3):
    """
    Full test eval with optional ensemble (majority vote).
    If agent picks fewer than min_picks stocks, fall back to buy-and-hold
    for that day to avoid unrealistic concentration.
    Returns diagnostics dict.
    """
    if not isinstance(q_nets, list):
        q_nets = [q_nets]

    dates = sorted(df_test['date'].unique())
    agent_d, bh_d = [], []
    prev_inv = set()
    diag = {'n_invested': [], 'n_total': []}

    for date in dates:
        dd = df_test[df_test['date'] == date]
        if len(dd) == 0:
            continue
        inv, all_r = [], []
        for _, row in dd.iterrows():
            feat = row[feature_cols].values.astype(float)
            pos = 1 if row['ticker'] in prev_inv else 0
            regime = row['regime']
            st_t = torch.FloatTensor(np.append(feat, [pos, regime])).unsqueeze(0)
            with torch.no_grad():
                votes = sum(n(st_t).argmax(1).item() for n in q_nets)
            if votes > len(q_nets) / 2:
                inv.append(row['ticker'])
            all_r.append(row['ret'])

        diag['n_invested'].append(len(inv))
        diag['n_total'].append(len(dd))

        if len(inv) >= min_picks:
            r = dd[dd['ticker'].isin(inv)]['ret']
            new = set(inv) - prev_inv
            cost = len(new) / len(inv) * tc
            agent_d.append(r.mean() - cost)
        else:
            agent_d.append(np.mean(all_r))

        bh_d.append(np.mean(all_r))
        prev_inv = set(inv) if len(inv) >= min_picks else set()

    return np.array(agent_d), np.array(bh_d), dates[:len(agent_d)], diag


def compute_benchmarks(df, tc=0.0005, window=5):
    """
    Paper section 4.1 benchmarks:
      - Momentum: buy stocks with positive avg return over last `window` days
      - Reversion: buy stocks with negative avg return over last `window` days
    Equal-weighted, transaction costs applied on position changes.
    """
    dates = sorted(df['date'].unique())
    mom_d, rev_d = [], []
    prev_mom, prev_rev = set(), set()

    for date in dates:
        dd = df[df['date'] == date]
        past_dates = sorted(df[df['date'] < date]['date'].unique())

        if len(past_dates) < window:
            mom_d.append(dd['ret'].mean())
            rev_d.append(dd['ret'].mean())
            prev_mom, prev_rev = set(), set()
            continue

        past = df[df['date'].isin(past_dates[-window:])]
        avg_past = past.groupby('ticker')['ret'].mean()

        mom_tickers = set(avg_past[avg_past > 0].index) & set(dd['ticker'])
        rev_tickers = set(avg_past[avg_past < 0].index) & set(dd['ticker'])

        if mom_tickers:
            new = mom_tickers - prev_mom
            cost = (len(new) / len(mom_tickers)) * tc
            mom_d.append(dd[dd['ticker'].isin(mom_tickers)]['ret'].mean() - cost)
        else:
            mom_d.append(dd['ret'].mean())

        if rev_tickers:
            new = rev_tickers - prev_rev
            cost = (len(new) / len(rev_tickers)) * tc
            rev_d.append(dd[dd['ticker'].isin(rev_tickers)]['ret'].mean() - cost)
        else:
            rev_d.append(dd['ret'].mean())

        prev_mom = mom_tickers
        prev_rev = rev_tickers

    return np.array(mom_d), np.array(rev_d)

def compute_metrics(daily_returns, bh_d=None, mom_d=None, rev_d=None, label=""):
    cr = np.prod(1 + daily_returns) - 1
    mean = np.mean(daily_returns) * 252
    std = np.std(daily_returns) * np.sqrt(252)
    sharpe = mean / std if std > 0 else 0.0
    cumulative = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    metrics = {
        'label': label,
        'cumulative_return': cr,
        'annualized_sharpe': sharpe,
        'max_drawdown': max_dd,
    }
    if bh_d is not None:
        metrics['win_rate_vs_bh'] = np.mean(daily_returns > bh_d)
    if mom_d is not None:
        metrics['win_rate_vs_mom'] = np.mean(daily_returns > mom_d)
    if rev_d is not None:
        metrics['win_rate_vs_rev'] = np.mean(daily_returns > rev_d)
    return metrics


def print_metrics_table(metrics_list):
    print(f"\n{'Strategy':<20} {'Cum. Return':>12} {'Sharpe':>8} {'Max DD':>10} {'WR vs BH':>10} {'WR vs Mom':>10} {'WR vs Rev':>10}")
    print("-" * 82)
    for m in metrics_list:
        wr_bh  = f"{m['win_rate_vs_bh']:>9.1%}"  if 'win_rate_vs_bh'  in m else f"{'N/A':>9}"
        wr_mom = f"{m['win_rate_vs_mom']:>9.1%}" if 'win_rate_vs_mom' in m else f"{'N/A':>9}"
        wr_rev = f"{m['win_rate_vs_rev']:>9.1%}" if 'win_rate_vs_rev' in m else f"{'N/A':>9}"
        print(f"{m['label']:<20} {m['cumulative_return']:>11.2%} {m['annualized_sharpe']:>8.2f} {m['max_drawdown']:>10.2%} {wr_bh} {wr_mom} {wr_rev}")


def plot_results(agent_d, bh_d, mom_d, rev_d, dates, tc_label):
    a_cum  = np.cumprod(1 + agent_d) - 1
    b_cum  = np.cumprod(1 + bh_d)   - 1
    mo_cum = np.cumprod(1 + mom_d)  - 1
    re_cum = np.cumprod(1 + rev_d)  - 1

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, a_cum  * 100, label='DQN Agent',  linewidth=2)
    ax.plot(dates, b_cum  * 100, label='Buy & Hold', linewidth=2, alpha=0.7)
    ax.plot(dates, mo_cum * 100, label='Momentum',   linewidth=2, alpha=0.7)
    ax.plot(dates, re_cum * 100, label='Reversion',  linewidth=2, alpha=0.7)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(f'DQN Agent vs Benchmarks (TC = {tc_label})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    fig.tight_layout()
    fig.savefig(f'../results/poc_v2_{tc_label}.png', dpi=150)
    plt.show()
    print(f"Agent CR:     {a_cum[-1]:+.2%}")
    print(f"Buy&Hold CR:  {b_cum[-1]:+.2%}")
    print(f"Momentum CR:  {mo_cum[-1]:+.2%}")
    print(f"Reversion CR: {re_cum[-1]:+.2%}")
    print(f"Excess vs BH: {a_cum[-1] - b_cum[-1]:+.2%}")


if __name__ == '__main__':
    print("=" * 55)
    print("PoC v2: Paper-Faithful DQN (Small Scale)")
    print("=" * 55)

    df_train, df_val, df_test, fcols, regime_weights = prepare_data()

    TC = 0.0005
    HIDDEN_DIMS = [32, 64, 128]  # paper: ensemble of 3
    ensemble = []

    for reward_fn, sharpe_lambda in [("base", 0.1), ("sharpe", 0.1)]:
        for hdim in HIDDEN_DIMS:
            print(f"\n{'-'*55}")
            print(f"Training agent: hidden_dim={hdim}, TC={TC*10000:.0f}bps, 500k steps")
            print(f"{'-'*55}")

            net, _ = train_agent(
                df_train, df_val, fcols,
                transaction_cost=TC,
                n_steps=500_000,
                hidden_dim=hdim,
                reward_fn=reward_fn,
                sharpe_lambda=sharpe_lambda,
                regime_weights=regime_weights,
            )
            ensemble.append(net)

            os.makedirs('../results', exist_ok=True)
            torch.save(net.state_dict(), f'../results/v2_qnet_{hdim}_5bps.pt')

    # Evaluate ensemble with diagnostics
    print(f"\n{'='*55}")
    print("Ensemble evaluation (majority vote, 3 agents)")
    print(f"{'='*55}")

    a_d, bh_d, dates, diag = evaluate_portfolio(df_test, ensemble, fcols, tc=TC)

    metrics = [
        compute_metrics(a_d, bh_d, mom_d, rev_d, label="DQN Agent"),
        compute_metrics(bh_d, label="Buy & Hold"),
        compute_metrics(mom_d, label="Momentum"),
        compute_metrics(rev_d, label="Reversion"),
    ]
    print_metrics_table(metrics)

    mom_d, rev_d = compute_benchmarks(df_test, tc=TC)
    plot_results(a_d, bh_d, mom_d, rev_d, dates, '5bps')

    # diagnostics
    print(f"\nPortfolio diagnostics (test period):")
    print(f"  Avg stocks picked per day: {np.mean(diag['n_invested']):.1f} / {np.mean(diag['n_total']):.1f}")
    print(f"  Min/Max picked: {np.min(diag['n_invested'])} / {np.max(diag['n_invested'])}")
    print(f"  Days with 0 picks (all cash): {sum(1 for n in diag['n_invested'] if n == 0)}")
    print(f"  Days with <5 picks: {sum(1 for n in diag['n_invested'] if 0 < n < 5)}")

    # Individual agent results
    print("\nIndividual agent results:")
    for hdim, net in zip(HIDDEN_DIMS, ensemble):
        ind_d, _, _, ind_diag = evaluate_portfolio(df_test, net, fcols, tc=TC)
        cr = np.cumprod(1 + ind_d)[-1] - 1
        avg_pick = np.mean(ind_diag['n_invested'])
        print(f"  hdim={hdim:>3d}: CR={cr:+.2%}, avg picks/day={avg_pick:.1f}")