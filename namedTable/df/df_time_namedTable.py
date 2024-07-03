# /usr/bin/python
# coding=utf-8
"""namedTable with time index called 'period'
"""

import datetime

import numpy as np
import pandas as pd

from demandcore.datastructures.time_period_tensor import TimePeriodTensor
from namedTable.df.base import DFBasenamedTable
from namedTable.df.slice_generators import CondensedRowSliceTensorGenerator, CondensedRowPeriodSlice
from namedTable.domains.df import TPTDFDomain, DFTimeDomain, DFDomain
from namedTable.utils.merge_utils import group_and_sort_by_period


class DFTimenamedTable(DFBasenamedTable):

    def __init__(self, df):

        super(DFTimenamedTable, self).__init__(df)
        # set the df index to be the domain index for consistency (period level gaps might have been filled)
        self._df.index = self.domain.coords_index

    @classmethod
    def unravel(cls, domain, data, data_column_names=None):
        if not isinstance(domain, DFTimeDomain):
            raise ValueError("DFTimenamedTable namedTables only supported with a DFTimeDomain.")

        return super(DFTimenamedTable, cls).unravel(domain, data, data_column_names=data_column_names)

    def _create_domain(self, *args, **kwargs):
        """
        overriding the method in DFBasenamedTable class to create DFTime domain
        Args:
            *args:
            **kwargs:

        Returns:
            A DFTimeDomain
        """
        return DFTimeDomain(*args, **kwargs)

    def groupby(self, by, agg_func='sum'):
        """
        Groups the namedTable by a coordinate(s) of the domain and applies agg_func to create a new namedTable.
        Args:
            by: coordinates in domain to group by for
            agg_func: function to use for aggregation (should be compatible to pandas groupby agrgegation function)

        """

        df = self._groupby(by, agg_func)

        return DFTimenamedTable(df) if DFTimeDomain.PERIOD_NAME in by else DFBasenamedTable(df)

    def append(self, other):
        return super(DFTimenamedTable, self)._append(self, other, DFTimenamedTable)

    def rename(self, domain_names_map=None, columns=None, inplace=False):
        """
        Same as DFBasenamedTable.rename() except 'period' domain name rename is
        not allowed. Renaming a name to 'period' is also not allowed if that
        name is present in this namedTable.
        """
        domain_names_map = domain_names_map or {}
        columns = columns or {}

        # NOTE: 'period' is protected domain name
        name = self.__class__.__name__

        if DFTimeDomain.PERIOD_NAME in domain_names_map:
            raise ValueError('Renaming "{}" domain name is not supported in {}.'.format(DFTimeDomain.PERIOD_NAME, name))

        for domain_name, value in domain_names_map.items():
            if value == DFTimeDomain.PERIOD_NAME and domain_name in self.domain.names:
                raise ValueError('Renaming domain name "{}" to "{}" is not supported in {}.'
                                 .format(domain_name, DFTimeDomain.PERIOD_NAME, name))

        for column, value in columns.items():
            if value == DFTimeDomain.PERIOD_NAME and column in self.data_column_names:
                raise ValueError('Renaming column "{}" to "{}" is not supported in {}.'
                                 .format(column, DFTimeDomain.PERIOD_NAME, name))

        return super(DFTimenamedTable, self).rename(domain_names_map=domain_names_map, columns=columns, inplace=inplace)

    def shift(self, n_periods=1):
        """
        shift slices along the time axis with name 'period'

        Args:
            n_periods: can be either positive or negative

        Returns:
            A DFTimenamedTable object (with different domain)

        """

        if DFTimeDomain.PERIOD_NAME != self.domain.names[-1]:
            raise ValueError("the last column of the index must have name 'period' that represents time")
        if n_periods == 0:
            return self
        elif n_periods > 0:
            i = self.df.index
            obs_mask = i.codes[2] < max(i.codes[2]) - n_periods + 1
            new_index = pd.MultiIndex(levels=[i.levels[0], i.levels[1], i.levels[2][n_periods:]],
                                      codes=[i.codes[0][obs_mask], i.codes[1][obs_mask], i.codes[2][obs_mask]],
                                      names=i.names)
            df_shifted = self.df[obs_mask]
            df_shifted.index = new_index
        elif n_periods < 0:
            n_periods = abs(n_periods)
            i = self.df.index
            obs_mask = i.codes[2] > min(i.codes[2]) + n_periods - 1
            new_index = pd.MultiIndex(levels=[i.levels[0], i.levels[1], i.levels[2][:(-n_periods)]],
                                      codes=[i.codes[0][obs_mask], i.codes[1][obs_mask],
                                              i.codes[2][obs_mask] - n_periods],
                                      names=i.names)
            df_shifted = self.df[obs_mask]
            df_shifted.index = new_index
        return self.__class__(df_shifted)

    def format_as_time_series_df(self, fill_zeros=False, **kwargs):
        """
        Returns the namedTable as a pandas DataFrame formatted as a (multivalued) time series.
        The index of the dataframe is the time periods; there is one column for every distinct
        combination of values in coordinates of the domain other than 'period'.
        For example, if the domain coordinates are ('row','col','period'), the resulting dataframe
        has one column for each ('row','col') combination.

        Returns: a pandas DataFrame

        """
        # TODO: Make sure multidimensional namedTable case is handeled properly

        df = self.df.squeeze('columns').unstack(level=DFTimeDomain.PERIOD_NAME).T

        mapper = lambda d: d.bop_datetime

        df.index = df.index.map(mapper).rename("ds")

        if fill_zeros:
            period_index = self.domain.get_coord_unique_values(DFTimeDomain.PERIOD_NAME, sorted=True)
            df = df.reindex(index=[mapper(p) for p in period_index], fill_value=0.)
            # reindex only fills value for periods not in df index. So we call
            # fillna to ensure any nans for periods in df index are filled.
            df = df.fillna(0.)

        return df

    def format_as_condensed_slice_time_series(self):

        """
        Returns a PeriodSlicer, which reformats the namedTable as a sequence of CondensedRowPeriodSlices, one per period.
        See df_time_namedTable.PeriodSlicer for more details.
        """

        return PeriodSlicer(self)

    @classmethod
    def from_condensed_slices(cls, slice_tensor_generator):
        """
        Constructs a DFTimenamedTable from a CondensedRowSliceTensorGenerator, typically constructed previously by
        format_as_condensed_slice_time_series().
        Args:
            slice_and_period_iterator (CondensedRowSliceTensorGenerator):

        Returns: a DFTimenamedTable

        """

        df = CondensedRowSliceTensorGenerator.to_pandas_df(slice_tensor_generator)
        # If the input slicer was originally created by a PeriodSlicer and the DFTimenamedTable which was used to create it
        # had only a period coordinate, then an artificial coordinate would be added to the slices.
        # The name of the coordinate is defined by PeriodSlicer.DIMENSIONLESS_COORD_NAME (see PeriodSlicer below).
        if df.index.names[0] == PeriodSlicer.DIMENSIONLESS_COORD_NAME:
            df.index = pd.MultiIndex(levels=df.index.levels[1:], codes=df.index.codes[1:], names=df.index.names[1:])

        return DFTimenamedTable(df)

    def filter_gaps(self):
        return self.__class__(remove_group_with_gaps(self.df))


    def filter_by_period_range(self, period, inplace=False):
        """
        finds items whose starting date is between bop and eop of period

        Args:
            period (Period):
            inplace:

        Returns:
            None if inplace, a new namedTable otherwise
        """
        period_level_index = self.df.index.names.index(DFTimeDomain.PERIOD_NAME)
        period_mask = np.array([p in period for p in self.df.index.levels[period_level_index]])
        df_mask = period_mask[self._df.index.codes[period_level_index]]
        tmp = self._df[df_mask]
        if inplace:
            self._df = tmp.copy()
            self.domain = self.domain.__class__(self._df.index)
            self._df.index = self.domain.coords_index
            return None
        else:
            return self.__class__(tmp)


class TPTDFnamedTable(DFTimenamedTable):

    def __init__(self, df):

        super(TPTDFnamedTable, self).__init__(df)
        if self.shape[1] > 1:
            raise ValueError("TPTDFnamedTable can't have multiple columns.")
        self.row_column_name = self.domain.names[0]
        self.col_column_name = self.domain.names[1]
        self.data_column_name = self.data_column_names[0]


    def _create_domain(self, *args, **kwargs):
        """
        overriding the method in DFTimenamedTable class to create TPTDF domain
        Args:
            *args:
            **kwargs:

        Returns:
            A TPTDF Domain
        """

        return TPTDFDomain(*args, **kwargs)

    def format_as_time_series_df(self, fill_zeros=False, **kwargs):
        """
        Returns the namedTable as a pandas DataFrame formatted as a (multivalued) time series.
        The index of the dataframe is the time periods; there is one column for every distinct
        combination of values in coordinates of the domain other than 'period'.
        For example, if the domain coordinates are ('row','col','period'), the resulting dataframe
        has one column for each ('row','col') combination.

        Returns: a pandas DataFrame

        """
        # TODO: Make sure multidimensional namedTable case is handeled properly

        df = self.df.squeeze('columns').unstack(level=DFTimeDomain.PERIOD_NAME).T

        mapper = lambda d: d.bop_datetime

        df.index = df.index.map(mapper).rename("ds")

        if fill_zeros:
            df = df.reindex([mapper(p) for p in self.domain.period_index], fill_value=0.)

        return df

    def to_tpt(self):
        return TimePeriodTensor.load_from_pandas_df(self.df,
                                                    row_column_name=self.domain.names[0],
                                                    col_column_name=self.domain.names[1])

    @classmethod
    def from_tpt(cls, tpt, row_column_name='row', col_column_name='col', data_column_name='data'):

        df = tpt.to_pandas_df(row_column_name=row_column_name,
                              col_column_name=col_column_name,
                              data_column_name=data_column_name,
                              period_column_name='period',
                              use_multi_index=True
                              )

        return TPTDFnamedTable(df)


# TODO: this seems unused, delete?
def _check_gaps(df, days_per_period=7):
    periods = df.index.get_level_values('period')
    d1 = min(periods)[0]
    d2 = max(periods)[0]
    d1 = datetime.datetime.strptime(str(d1), '%Y%m%d')
    d2 = datetime.datetime.strptime(str(d2), '%Y%m%d')
    periods_from_timestamp = (d2 - d1).total_seconds() / (24.0 * 3600) / days_per_period + 1
    periods_from_counts = len(periods)
    return periods_from_timestamp == periods_from_counts


# more efficient implementation of check_gaps
def check_gaps_series(s):
    """
    return True if there is gap, else return False
    Args:
        s:

    Returns:

    """
    idx = np.argwhere((~np.isnan(s)).values).flatten()
    if len(idx) == 1:
        return s
    if ~np.all(np.diff(idx) == 1):
        return pd.Series(np.repeat(np.nan, len(s)), index=s.index)
    return s


def remove_group_with_gaps(df):
    return df.squeeze().unstack().apply(check_gaps_series, axis=1).stack().to_frame(name=df.columns[0])


class PeriodSlicer(object):
    """
    Class to construct time slices of DFTimenamedTables, where each slice is a matrix of dimensions
    <unique domain items (excluding period coord)> x <data columns>. Because namedTables can have sparse
    data, when constructing the matrices we need to explicitly distinguish elements that are 0 (and might
    be sparsely represented in the namedTable) from elements that are not present in the namedTable.

    For exmaple, the namedTable
                                data
        ('p1','s1','period1') | 1 1
        ('p1','s2','period1') | 0 3
        ('p1','s1','period2') | 2 0
        ('p1','s2','period2') | 4 0
        ('p2','s1','period3') | 0 5

    gets converted into a the tensor (each matrix has been transposed for display purposes)

             ('p1','s1') | ('p1','s2') | ('p2', 's1')
        data      1            0            *           [period1]
                  1            3            *

             ('p1','s1') | ('p1','s2') | ('p2', 's1')
        data      2            4            *           [period2]
                  0            0            *


             ('p1','s1') | ('p1','s2') | ('p2', 's1')
        data      *            *            0           [period3]
                  *            *            5

    Each of the slices is represented by an instance of the class CondensedRowPeriodSlice
    """

    DIMENSIONLESS_COORD_NAME = '_PeriodSlicer::DIMENSIONLESS_COORD_NAME_'
    DIMENSIONLESS_COORD_VALUE = '_PeriodSlicer::DIMENSIONLESS_COORD_VALUE_'

    def __init__(self, df_time_namedTable):

        if not isinstance(df_time_namedTable, DFTimenamedTable):
            raise ValueError('PeriodSlicer can only be created with DFTimenamedTables')

        if len(df_time_namedTable.domain.names) == 1:  # only period coordinate
            domain = DFDomain.cartesian_product(DFDomain([self.DIMENSIONLESS_COORD_VALUE],
                                                         names=self.DIMENSIONLESS_COORD_NAME),
                                                df_time_namedTable.domain)
            domain = DFTimeDomain.from_DFDomain(domain)
        else:
            domain = df_time_namedTable.domain

        self._group_codes, self._sorted_coords, self._period_starts, group_rep_idxs = \
            group_and_sort_by_period(domain)

        self.row_index = self._drop_last_level(domain.coords_index[group_rep_idxs])
        self.col_names = df_time_namedTable.data_column_names
        self.period_index = domain.coords_index.levels[-1]
        self._df_mat = df_time_namedTable._df_mat

    @staticmethod
    def _drop_last_level(multiindex):
        # implement a drop level function to make sure the output is a multiindex (pandas returns an index
        # if there is only one level)
        return pd.MultiIndex(levels=multiindex.levels[:-1], codes=multiindex.codes[:-1], names=multiindex.names[:-1])

    def get_slice_generator(self, valid_period_indices=None):
        """
        Returns an CondensedRowSliceTensorGenerator, which generates CondensedRowPeriodSlices.
        All of the CondensedRowPeriodSlices share the same row and col valid_period_indices.
        A slice is returned in order for every period in the df_time_namedTable from which this slicer was created
        (including periods of no actual data).

        Args:
            valid_period_indices (iterable of int): if defined, determines which period indices will be generated

        """

        return CondensedRowSliceTensorGenerator(self._get_slice_generator(valid_period_indices))

    def _get_slice_generator(self, valid_period_indices=None):
        if valid_period_indices is None:
            valid_period_indices = set(range(len(self._period_starts)))

        if not set(valid_period_indices) <= set(range(len(self._period_starts))):
            raise ValueError('indices needs to be a subset of range(%s)' % len(self._period_starts))

        for k in range(len(self._period_starts)):

            if k not in valid_period_indices:
                continue

            if k+1 < len(self._period_starts):
                slice_idxs = self._sorted_coords[self._period_starts[k]:self._period_starts[k+1]]
            else:
                slice_idxs = self._sorted_coords[self._period_starts[k]:]

            retype = lambda x, t: x.astype(t) if x.dtype != t else x
            slice = retype(self._df_mat[slice_idxs, :], self._df_mat.dtype)

            mask = np.full(len(self.row_index), False, dtype=bool)
            mask[self._group_codes[slice_idxs]] = True

            yield CondensedRowPeriodSlice(slice, self.row_index, self.col_names, mask, self.period_index[k])
