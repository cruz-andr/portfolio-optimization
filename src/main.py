import os
import numpy as np
import torch

from pipeline import prepare_data
from train import train_agent
from evaluate import evaluate_portfolio, compute_benchmarks, compute_metrics, print_metrics_table, plot_results


print("=" * 55)
print("PoC v2: Paper-Faithful DQN (Small Scale)")
print("=" * 55)

TC = 0.0005
HIDDEN_DIMS = [32, 64, 128]
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.parquet')

CONFIGS = [
(False, "base",          0.1),
(False, "sharpe",        0.1),
(False, "sharpe+regime", 0.1),
(True,  "sharpe+regime", 0.1),
]

for use_factors, reward_fn, sharpe_lambda in CONFIGS:
    factor_label = "factors" if use_factors else "no_factors"
    print(f"\n{'='*55}")
    print(f"Run: {factor_label}")
    print(f"{'='*55}")

    df_train, df_val, df_test, fcols, regime_weights = prepare_data(
        parquet_path=DATA_PATH, use_factors=use_factors
    )
    ensemble = []
    for hdim in HIDDEN_DIMS:
        print(f"\n{'-'*55}")
        print(f"Training: {factor_label} | {reward_fn} | hidden_dim={hdim}")
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

        os.makedirs('results', exist_ok=True)
        torch.save(net.state_dict(), f'results/qnet_{factor_label}_{reward_fn}_{hdim}.pt')

    # evaluate this config
    a_d, bh_d, dates, diag = evaluate_portfolio(df_test, ensemble, fcols, tc=TC)
    mom_d, rev_d = compute_benchmarks(df_test, tc=TC)

    print(f"\nResults: {factor_label} | {reward_fn}")
    metrics = [
        compute_metrics(a_d, bh_d=bh_d, mom_d=mom_d, rev_d=rev_d, label="DQN Agent"),
        compute_metrics(bh_d, label="Buy & Hold"),
        compute_metrics(mom_d, label="Momentum"),
        compute_metrics(rev_d, label="Reversion"),
    ]
    print_metrics_table(metrics)
    plot_results(a_d, bh_d, mom_d, rev_d, dates, f'{factor_label}_{reward_fn}')

    print(f"\nPortfolio diagnostics:")
    print(f"  Avg stocks picked: {np.mean(diag['n_invested']):.1f} / {np.mean(diag['n_total']):.1f}")
    print(f"  Days with 0 picks: {sum(1 for n in diag['n_invested'] if n == 0)}")
    print(f"  Days with <5 picks: {sum(1 for n in diag['n_invested'] if 0 < n < 5)}")