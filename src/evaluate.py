import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    fig.savefig(f'results/{tc_label}.png', dpi=150)
    plt.show()
    print(f"Agent CR:     {a_cum[-1]:+.2%}")
    print(f"Buy&Hold CR:  {b_cum[-1]:+.2%}")
    print(f"Momentum CR:  {mo_cum[-1]:+.2%}")
    print(f"Reversion CR: {re_cum[-1]:+.2%}")
    print(f"Excess vs BH: {a_cum[-1] - b_cum[-1]:+.2%}")


