# /usr/bin/python
"""namedTable building utilities"""
from copy import copy

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.preprocessing
import logging
from itertools import chain

from scipy.sparse import coo_matrix, csc_matrix

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def standardize_matrix(m, percentile=95, scale=False, max_value=None):
    """
    Clip matrix to a certain max percentile. Convert negative values to 0.

    Scale to 1 std if scale == True. (Will not 0 mean
    because that would ruin sparsitiy.)
    Args:
        m (scipy.sparse.spmatrix)
        percentile (float): float between [0, 100]. See np.percentile for more details. (Default is 95.)
        scale (bool): If true, scale to unit standard deviation. Will not center the matrix
            as that would ruin sparsity. (Default is False.)
        max_value (float): max value to use if not using percentile

    Returns:
        standard_m (scipy.sparse.coo_matrix)

    Raises:
        ValueError if m is not of the correct type
    """
    if not isinstance(m, scipy.sparse.spmatrix):
        raise ValueError("m must be a scipy.sparse.spmatrix matrix.")
    m = m.tocsc()
    m[m < 0] = 0.
    m = m.tocoo()
    if max_value is not None:
        new_max = max_value
    else:
        new_max = np.percentile(m.data, percentile, method='higher')
    np.clip(m.data, None, new_max, out=m.data)
    if scale:
        sklearn.preprocessing.scale(m.data, with_mean=False, copy=False)
    return scipy.sparse.coo_matrix((m.data, (m.row, m.col)), shape=m.shape)


def rolling_mean(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate([np.repeat(np.nan, n - 1), ret[n - 1:] / n])


def rolling_sum(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate([np.repeat(np.nan, n - 1), ret[n - 1:]])


def rolling_sum_series(x, n=3):
    a = x.values
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    vals = np.concatenate([np.repeat(np.nan, n - 1), ret[n - 1:]])
    return pd.Series(vals, index=x.index)


def rolling_sum_df(x, n=3):
    a = x.T.values[0]
    idx = x.index.get_level_values('period')
    if n - 1 > len(a):
        vals = np.repeat(np.nan, len(a))
        return pd.Series(vals, index=idx)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    vals = np.concatenate([np.repeat(np.nan, n - 1), ret[n - 1:]])
    return pd.Series(vals, index=idx).to_frame(name=x.columns[0])


def _get_full_period_index(idx):
    tmp = idx.copy()
    min_label = min(tmp.codes[-1])
    max_label = max(tmp.codes[-1])
    my_len = max_label - min_label + 1
    new_codes = [[list(x)[0]] * my_len for x in tmp.codes[:-1]]
    new_codes.append([x for x in range(min_label, max_label + 1)])
    return tmp.set_codes(new_codes, verify_integrity=False)


def _get_full_period_by_permutation(df):
    return df.reindex(pd.MultiIndex.from_product(df.index.levels), fill_value=np.nan)


def _get_full_period_by_group(df):
    g = df.groupby(df.index.names[:-1])
    return g.apply(
        lambda x: x.reindex(_get_full_period_index(x.index), fill_value=np.nan).reset_index(level=x.index.names[:-1],
                                                                                            drop=True))


def cum_sum_shifting(df, n_periods=6, agg_func='sum', ignore_na=False):
    df = df.copy()
    non_period_coords = df.index.names[:-1]

    for week_index in range(1, abs(n_periods)):
        shift_index = week_index if n_periods > 0 else -week_index
        df[week_index] = df.groupby(non_period_coords)[df.columns[0]].shift(shift_index)
    if agg_func == 'sum':
        res = df.sum(axis=1, skipna=ignore_na)
    elif agg_func == 'mean':
        res = df.mean(axis=1, skipna=ignore_na)
    else:
        raise ValueError("Only 'sum' and 'mean' are supported as agg_func")
    return res.to_frame(name=df.columns[0])


def fill_gaps_merge(df):
    df1 = df.reset_index()
    df1.drop_duplicates(['row', 'col'])
    full_period_list = df.index.levels[-1]
    df2 = df1.drop_duplicates(['row', 'col'])
    df3 = df2[['row', 'col']]
    full_period_df = full_period_list.to_frame()
    full_period_df = full_period_df.reset_index(drop=True)
    df3['tmp'] = 1
    full_period_df['tmp'] = 1
    df_merge = pd.merge(df3, full_period_df, on=['tmp'])
    del df_merge['tmp']
    return df_merge.merge(df.reset_index(), how='left', left_on=['row', 'col', 'period'],
                          right_on=['row', 'col', 'period']).set_index(['row', 'col', 'period'])


def fill_gaps_with_na_series(s):
    vals = s.values.flatten()
    idx1 = s.index.codes[-1]
    m = min(idx1)
    n = max(idx1)
    res = np.repeat(np.nan, n - m + 1)
    for i, x in enumerate(idx1):
        res[x - m] = vals[i]
    return pd.Series(res, index=s.index.levels[-1][m:(n + 1)])


def fill_gaps_with_na_df(df):
    g = df.groupby(df.index.names[:-1])
    return g.apply(fill_gaps_with_na_series).to_frame(name=df.columns[0])


def cum_sum_merge(df, n_periods=2):
    return cum_sum_shifting(fill_gaps_merge(df)
                            , n_periods=n_periods
                            ).dropna()


def rolling_func_unstack(df, func, lb=0, ub=0, min_periods='all', full_window_only=True):
    assert isinstance(min_periods, (int,)) or min_periods == 'all', "min_periods must be an integer or 'all'"
    if min_periods == 'all' and not full_window_only:
        logger.warning("min_periods is 'all' but full_window_only is False. full_window_only will be treated as True.")

    if np.issubdtype(df.values.dtype, np.integer):
        df = df.astype(float, copy=False)

    if min_periods == 'all':
        min_periods = None
    unstacked_df = df.squeeze().unstack('period')
    lb = -unstacked_df.shape[1]+1 if lb is None else lb
    ub = unstacked_df.shape[1]-1 if ub is None else ub

    n_periods = ub - lb + 1
    shift = -ub
    add_fake_columns = 0 < (min_periods or 0) and not full_window_only
    if add_fake_columns:
        # Add nan columns to the right
        fake_columns = ['__fake_column_%i'%i for i in range(ub)]
        unstacked_df = pd.concat([unstacked_df,
                                  pd.DataFrame(np.nan,
                                  index=unstacked_df.index,
                                  columns=pd.Index(fake_columns,name='period')),
                                 ],
                                 sort=True,
                                 axis=1)

    out_df = func(unstacked_df.rolling(n_periods, axis=1, min_periods=min_periods)).transpose().\
                  shift(shift).transpose()

    if full_window_only and lb < 0:
        # Remove columns on the left
        out_df.drop(columns=out_df.columns[:abs(lb)], inplace=True)

    if add_fake_columns:
        out_df.drop(fake_columns, axis=1, inplace=True)
    return out_df.stack().to_frame(name=df.columns[0])


def rolling_unstack(df_or_tpt, lb=0, ub=0, min_periods='all', full_window_only=True,
                    func ='sum', method='pandas'):
    if func not in ['sum', 'mean']:
        raise ValueError("func needs to be 'sum' or 'mean'.")
    assert isinstance(min_periods, (int,)) or min_periods == 'all', \
        "min_periods must be an integer or 'all'"
    if min_periods == 'all' and not full_window_only:
        logger.warning("min_periods is 'all' but full_window_only is False. full_window_only will be treated as True.")

    if method == 'pandas':
        if func == 'sum':
            func = (lambda x: x.sum()) if min_periods != 'all' else (lambda x: x.apply(np.nansum, raw=True))
        else:  # func == 'mean'
            func = (lambda x: x.mean()) if min_periods != 'all' else (lambda x: x.apply(np.nanmean, raw=True))

        return rolling_func_unstack(df_or_tpt, func=func, lb=lb, ub=ub, min_periods=min_periods,
                                    full_window_only=full_window_only)
    elif method == 'tpt':
        return rolling_func_unstack_tpt(df_or_tpt, lb, ub,
                                        min_periods=min_periods,
                                        full_window_only=full_window_only,
                                        func=func)
    else:
        raise ValueError("methods needs to be 'pandas' or 'tpt'.")


def cum_sum_unstack_series(s):
    s = s.copy()
    s[s.first_valid_index():] = s[s.first_valid_index():].cumsum(skipna=False)
    return s


def cum_sum_unstack(df):
    return df.squeeze().unstack('period').apply(cum_sum_unstack_series, axis=1).stack().to_frame(name=df.columns[0])


def cum_mean_unstack_series(s):
    s = s.copy()
    tmp = s[s.first_valid_index():]
    s[s.first_valid_index():] = tmp.cumsum(skipna=False) / pd.Series(np.arange(1, len(tmp) + 1), tmp.index)
    return s


def cum_mean_unstack(df):
    return df.squeeze().unstack('period').apply(cum_mean_unstack_series, axis=1).stack().to_frame(name=df.columns[0])


def cum_mean_ignore_leading_nan(arr):
    res = arr.copy()
    idx = np.argwhere(~np.isnan(res))[0][0]
    tmp = np.cumsum(res[idx:])
    res[idx:] = tmp / np.arange(1, len(tmp) + 1)
    return res


def cum_mean_ignore_leading_nan_df(df):
    tmp = df.squeeze().unstack('period')
    arr = tmp.values
    idx = tmp.index
    col_names = tmp.columns
    return pd.DataFrame(np.apply_along_axis(cum_mean_ignore_leading_nan, 1, arr),
                        index=idx,
                        columns=col_names).stack().to_frame(name=df.columns[0])


def cum_sum_ignore_leading_nan(arr):
    res = arr.copy()
    idx = np.argwhere(~np.isnan(res))[0][0]
    tmp = np.cumsum(res[idx:])
    res[idx:] = tmp
    return res


def cum_sum_ignore_leading_nan_df(df):
    tmp = df.squeeze().unstack('period')
    arr = tmp.values
    idx = tmp.index
    col_names = tmp.columns
    return pd.DataFrame(np.apply_along_axis(cum_sum_ignore_leading_nan, 1, arr),
                        index=idx,
                        columns=col_names).stack().to_frame(name=df.columns[0])


