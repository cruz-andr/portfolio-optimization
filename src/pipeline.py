import pandas as pd
import numpy as np

from regime import prepare_regimes
from factors import add_factor_ranks, FACTOR_COLS


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


def prepare_data(parquet_path='../data/stock_data.parquet', use_factors=True):
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


    if use_factors:
        df_train = add_factor_ranks(df_train)
        df_val   = add_factor_ranks(df_val)
        df_test  = add_factor_ranks(df_test)
        feature_cols = FEATURE_COLS 
    else:
        feature_cols = TECHNICAL_COLS + FUNDAMENTAL_COLS + PRICE_COLS

    # paper eq. 3: cash reward = cross-sectional mean return
    for split in [df_train, df_val, df_test]:
        split['cross_sect_mean_ret'] = split.groupby('date')['ret'].transform('mean')
        
    _, df_train, df_val, df_test, regime_weights = prepare_regimes(df_train, df_val, df_test)

    print(f"State dim: {len(FEATURE_COLS)} features + 1 dummy = {len(FEATURE_COLS)+1}")
    return df_train, df_val, df_test, feature_cols, regime_weights
