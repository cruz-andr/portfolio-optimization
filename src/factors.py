import pandas as pd
import numpy as np


def add_factor_ranks(df):
    """
    Add cross-sectional percentile ranks for four Fama-French inspired factors.
    Ranks are computed per date across all stocks — so rank=1.0 means top stock
    on that day, rank=0.0 means bottom stock.

    Factors added:
      - rank_size       : market cap rank (SMB proxy)
      - rank_value      : book-to-market rank (HML proxy)
      - rank_momentum   : sma_5 / sma_200 rank (UMD proxy)
      - rank_quality    : mean(roe, roa) rank (RMW proxy)
    """
    df = df.copy()

    # momentum signal: ratio of short-term to long-term moving average
    df['_mom_signal'] = df['sma_5'] / df['sma_200'].replace(0, np.nan)

    # quality signal: average of roe and roa
    df['_quality_signal'] = df[['roe', 'roa']].mean(axis=1)

    # compute percentile ranks cross-sectionally per date
    for col, rank_col in [
        ('mkt_cap',          'rank_size'),
        ('book_to_market',   'rank_value'),
        ('_mom_signal',      'rank_momentum'),
        ('_quality_signal',  'rank_quality'),
    ]:
        df[rank_col] = (
            df.groupby('date')[col]
            .rank(pct=True, na_option='keep')
        )

    for col in ['rank_size', 'rank_value', 'rank_momentum', 'rank_quality']:
        df[col] = df[col].fillna(0.5)

    df = df.drop(columns=['_mom_signal', '_quality_signal'])

    return df


FACTOR_COLS = ['rank_size', 'rank_value', 'rank_momentum', 'rank_quality']