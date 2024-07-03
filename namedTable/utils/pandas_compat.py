"""
This modules provides functionality to manage sparse DataFrame in pandas 0.24+ and 1+.
It provides an abstraction of a SparseDataFrame based on the pandas version. In pandas 0.24 it's simply a
pd.SparseDataframe. In pandas 1+ this was deprecated, so it's defined as a DataFrame that has all sparse columns.
"""

import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scipy.sparse

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PANDAS_MAJOR_VERSION = int(pd.__version__.split('.')[0])


def is_SparseDataFrame(df):
    if PANDAS_MAJOR_VERSION == 0:
        return isinstance(df, pd.SparseDataFrame)
    else:
        try:
            # Following scikit-learn's example https://github.com/scikit-learn/scikit-learn/pull/16728
            # pandas doesn't have a single method to check all columns are sparse: https://github.com/pandas-dev/pandas/issues/26706#issuecomment-604983887
            # Assumption: if .sparse exists, all of df's columns are sparse dtype. Somewhat enforced by
            # DFBasenamedTable's _df_is_sparse() method.
            # Ordinary df with dense data do not have the .sparse accessor.
            return hasattr(df, 'sparse')
        except Exception as e:
            # Fallback check. n_sparse_columns is rather slow compared to hasattr.
            logger.warning('hasattr check failed somehow, using slower n_sparse_columns check. Reason: %r', e)
            return n_sparse_columns(df) == df.shape[1]


def SparseDataFrame(mat_or_df, columns=None, index=None):
    """
    This function should be used to create sparse DataFrame in namedTable and other modules that use namedTable.
    In pandas >=1.x a SparseDataFrame is defined as a DataFrame with all sparse columns.
    Note: if mat_or_df is already a SparseDataFrame then the method returns mat_or_df itself.

    Args:
        mat_or_df (pd.DataFrame, sparse matrix or numpy array):
        columns: if mat_or_df is not a DataFrame, defines columns of the output DataFrame
        index: if mat_or_df is not a DataFrame, defines index of the output DataFrame

    Returns:
        pandas DataFrame with sparse columns if pandas >=1.x, otherwise a pandas SparseDataFrame.
    """
    if not isinstance(mat_or_df, (pd.DataFrame, np.ndarray)) and not scipy.sparse.issparse(mat_or_df):
        raise ValueError("mat_or_df needs to ba a dataframe, numpy array or sparse matrix")
    if isinstance(mat_or_df, pd.DataFrame):
        if is_SparseDataFrame(mat_or_df):
            return mat_or_df
        else:
            mat_or_df, columns, index = mat_or_df.values, mat_or_df.columns, mat_or_df.index
    if scipy.sparse.issparse(mat_or_df):
        mat = mat_or_df
    else:
        mat = scipy.sparse.csc_matrix(mat_or_df)

    if PANDAS_MAJOR_VERSION == 0:
        df = pd.SparseDataFrame(mat,
                                columns=columns,
                                index=index,
                                default_fill_value=0,
                                dtype=mat.dtype)

    else:
        df = pd.DataFrame.sparse.from_spmatrix(mat,
                                               columns=columns,
                                               index=index)
    return df


def n_sparse_columns(df):
    if PANDAS_MAJOR_VERSION == 0:
        # TODO: Review this case
        return df.shape[1] if isinstance(df, pd.SparseDataFrame) else 0
    else:
        return sum([pd.api.types.is_sparse(ct) for ct in df.dtypes])


def is_numeric(df):
    return all([is_numeric_dtype(t) for t in df.dtypes])


def to_coo(df):
    if PANDAS_MAJOR_VERSION == 0:
        return df.to_coo()
    else:
        df = _to_coo_replace_columns_if_dtypes_fails(df)
        return df.sparse.to_coo()


def _to_coo_replace_columns_if_dtypes_fails(df):
    """
    Pandas 1.x seems to fail when accessing dtypes if the columns have a level of type IntervalIndex.
    The reason for this seems to be that they try to access the first element of dtypes as dtypes[0], but if "0" is
    interpreted as a cardinal value to lookup in the intervals of the index as opposed to a positional value.
    See test_pandas_compat.py:test_to_coo_with_range_index_columns for examples.
    """
    try:
        df.dtypes[0]  # if this fails, catch the excepction and try replacing the columns, which are not relevant for to_coo()
        return df
    except KeyError:
        pass

    columns = df.columns
    df = df.copy(deep=False)
    df.columns = pd.Index(np.arange(len(columns)))
    return df


def df_iloc(df, row_slice=None, col_slice=None):
    row_slice = np.arange(df.shape[0]) if row_slice is None else row_slice
    col_slice = np.arange(df.shape[1]) if col_slice is None else col_slice

    return df.iloc[row_slice, col_slice]
