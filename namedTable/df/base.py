# /usr/bin/python
# coding=utf-8
import logging

import numpy as np
import pandas as pd
import scipy.sparse

from namedTable.base import namedTable
from namedTable.domains.df import DFDomain, DFDomainCreationError
from namedTable.exceptions import ItemNotFoundException
from namedTable.utils import merge_utils
import namedTable.utils.pandas_compat as pd_compat

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DFBasenamedTable(namedTable):
    """
    namedTable backed by pandas DataFrame.
    """

    def __init__(self, df):
        """
        **kwargs are left out intentionally because it might cause ambiguity when
        creating namedTables using self.__class__() methods. This forces the subclasses
        to override __init__() if necessary

        Args:
            df (Pandas DataFrame/SparseDataFrame): the dataframe to create the namedTable. The index of the dataframe
                                                   will be used to create the domain, and the columns will be the data
                                                   columns of the namedTable.
           NOTES on version 3.x:
                * 'domain_columns' and 'data_column_names' are no longer arguments and should instead
                                be formatted into df by the caller.
                * The data columns of namedTables (and the input columns of df) are now required to be numerical and an
                  error will be raised otherwise.
           """
        if df.isnull().values.any():
            raise ValueError('nans in input df not allowed.')

        if not pd_compat.is_numeric(df):
            raise ValueError('Input dataframe needs to have only numerical values.')

        self._df = df

        domain = self._create_domain(self._df.index, copy=False)

        self.data_column_names = df.columns
        self._df.index = domain.coords_index

        super(DFBasenamedTable, self).__init__(domain)

    @property
    def df(self):
        return self._df

    @classmethod
    def unravel(cls, domain, data, data_column_names=None):
        if not isinstance(domain, DFDomain):
            raise ValueError("DFnamedTable namedTables only supported with a DFDomain.")

        if data_column_names is None:
            if len(data.shape) == 1 or data.shape[1] == 1:
                data_column_names = ['data']
            else:
                data_column_names = ['data_col_%i' % i for i in range(data.shape[1])]

        if scipy.sparse.issparse(data):
            df = pd_compat.SparseDataFrame(data,  index=domain.coords_index, columns=data_column_names)
        else:
            df = pd.DataFrame(data, index=domain.coords_index, columns=data_column_names)

        return cls(df)

    def _create_domain(self, *args, **kwargs):
        """
        This is the function that is used in init() to create domains specific to the class
        override as necessary in subclasses

        Args:
            *args: arguments for domain constructor
            **kwargs: arguments for domain constructor

        Returns:
            A domain corresponding to the class

        """
        return DFDomain(*args, **kwargs)

    @property
    def data_column_names(self):
        return self.df.columns

    @data_column_names.setter
    def data_column_names(self, data_column_names):
        if data_column_names is not None:
            # TODO: should be a ValueError instead of assert.
            assert isinstance(data_column_names, (str, list, pd.Index)), \
                'data_column_names needs to be str, list(str) or pd.Index'

            if isinstance(data_column_names, (str,)):
                data_column_names = [data_column_names]

        if data_column_names is not None:
            if len(self._df.columns) != len(data_column_names):
                raise ValueError('len(data_column_names) should match number of non-domain df columns')
            self._df.columns = data_column_names

    @classmethod
    def _df_is_sparse(cls, df):
        n_sparse_columns = pd_compat.n_sparse_columns(df)
        if 0 < n_sparse_columns < df.shape[1]:
            raise ValueError("DataFrame to create a namedTable should be either all sparse or all dense.")
        return n_sparse_columns == df.shape[1]

    @property
    def shape(self):
        return self.df.shape

    @property
    def sparse(self):
        # A namedTable is either sparse (all columns of internal df are sparse) or dense.
        if not hasattr(self, "_sparse"):
            # This can happen if we load a namedTable created with a prior version, where
            # the sparse attribute was not yet defined. This ensures backwards-
            # compatibility in that case.
            self._sparse = self._df_is_sparse(self.df)
        return self._sparse

    @property
    def _df_mat(self):
        """
        Returns the namedTable as a matrix; a np array or a sparse matrix depending on whether the namedTable is
        non-sparse or sparse.
        The property is "private" so it's used only by "friend" classes, because it produces
        aliasing and/or memory overhead.
        """
        if self.sparse:
            df_mat = pd_compat.to_coo(self.df).tocsr()  # NOTE: unclear if this produces a lot of memory overhead.
        else:
            df_mat = self.df.values
        return df_mat

    def _get(self, item):

        return self.df.loc[item][0]

    def ravel(self, ravel_domain=None, fill_undefined=None, sparse=False, except_if_undefined=True):
        """
        Ravels the namedTable.

        Args:
            ravel_domain (DFDomain): If != None use this domain for raveling.
            fill_undefined (Numerical): If ravel_domain != None, value to fill on elements in ravel_domain not present
                                        in namedTable. Will raise ItemNotFoundException if fill_undefined = None and
                                        elements of ravel_domain are not in namedTable.
            sparse (bool): If True return a coo_matrix.
            except_if_undefined (bool): If True, raise ReindexException exception when fill_undefined=None and some
                                        elements of ravel_domain are not in the namedTable.
                                        If False, return an array and a boolean mask with indicating the elements in
                                        ravel_domain that could be raveled.
                                        If fill_undefined is not None, this flag is ignored and no mask is returned.
        """
        if fill_undefined is not None:
            if (not isinstance(fill_undefined, (int, float))) or np.isnan(fill_undefined):
                raise ValueError('fill_undefined needs to be a number or infinity.')
            if except_if_undefined:
                logger.warning("except_if_undefined is set with fill_undefined. fill_undefined will take precedence.")

        def to_return_array(df):
            if sparse:
                return pd_compat.to_coo(df) if self.sparse else scipy.sparse.coo_matrix(df.values)
            else:
                return df.values.reshape(-1, 1) if df.shape[1] == 1 else df.values

        if ravel_domain is None:
            return to_return_array(self.df)
        elif len(ravel_domain.coords_index) == 0:
            df = pd.DataFrame(index=[], columns=self.data_column_names, dtype="float")
            mask = np.array([], dtype=bool)
            if except_if_undefined or fill_undefined is not None:
                return to_return_array(df)
            else:
                return to_return_array(df), mask
        else:
            if fill_undefined is None:
                if except_if_undefined:
                    try:
                        ravel_df = merge_utils.reindex(self.df, ravel_domain.coords_index, fill_na=None,
                                                       except_on_na=True)
                        return to_return_array(ravel_df)
                    except merge_utils.ReindexException:
                        raise ItemNotFoundException('Could not ravel on ravel_domain with except_if_undefined=True without '
                                                    'fill_undefined. Some items in ravel_domain were not found')
                else:
                    ravel_df, mask = merge_utils.reindex(self.df, ravel_domain.coords_index, fill_na=None,
                                                         except_on_na=False)
                    return to_return_array(ravel_df), mask

            else:
                return to_return_array(merge_utils.reindex(self.df, ravel_domain.coords_index, fill_na=fill_undefined,
                                                           except_on_na=False))

    def reindex(self, domain, fill_undefined=None, inplace=False, except_if_undefined=True):
        """
        Reindex the data based on a new domain.

        Args:
            domain (DFDomain): domain to reindex to domain_mapping (Callable DFDomain->DFDomain): If != None then
                                namedTable is going to be reindex based don domain_map(domain)
            fill_undefined (bool): value to fill on elements in domain (or domain_map(domain) if domain_map != None)
                                    not present in namedTable. Will raise ItemNotFoundException if fill_undefined = None
                                    and elements of ravel_domain are not in namedTable.
            inplace (bool): If True, modifies the namedTable, if not returns a new one.
            except_if_undefined (bool): If True, raise ReindexException exception when fill_undefined=None and some
                                        elements of domain are not in the namedTable.
                                        If False, reindex the namedTable (or return a new one) and return boolean mask
                                        indicating the elements in domain that where founf on the namedTable's domain.
                                        If fill_undefined is not None, this flag is ignored and no mask is returned.
                                        If inplace=True no mask is returned, even if except_if_undefined=True
        """
        if fill_undefined is not None:
            if (not isinstance(fill_undefined, (int, float))) or np.isnan(fill_undefined):
                raise ValueError('fill_undefined needs to be a number or infinity.')

        if len(domain.coords_index) == 0:
            # no data, same index
            df = pd.DataFrame(index=domain.coords_index, columns=self.data_column_names, dtype=np.float64)
            mask = np.array([], dtype=bool)
        else:
            if fill_undefined is None:
                if except_if_undefined:
                    try:
                        df = merge_utils.reindex(self.df, domain.coords_index, fill_na=None, except_on_na=True)
                    except merge_utils.ReindexException:
                        raise ItemNotFoundException("If domain contains values undefined in namedTable, fill_undefined "
                                                    "needs to be not None.")
                else:
                    df, mask = merge_utils.reindex(self.df, domain.coords_index, fill_na=None, except_on_na=False)
            else:
                df = merge_utils.reindex(self.df, domain.coords_index, fill_na=fill_undefined, except_on_na=False)

        new_namedTable = self.__class__(df)
        if inplace:
            self._df = new_namedTable.df
            self.domain = new_namedTable.domain
        else:
            if not except_if_undefined:
                return new_namedTable, mask
            return new_namedTable

    def empty(self, inplace=False, dtype="float"):
        """
        Returns an empty namedTable with the same coordinates as this namedTable
        Args:
            inplace (bool):

        Returns:

        """

        df = pd.DataFrame([], columns=self.data_column_names, dtype=dtype)
        empty_domain = self.domain.__class__([], names=self.domain.names)
        df.index = empty_domain.coords_index
        if inplace:
            self._df = df
            self.domain = empty_domain
            self._df.index = empty_domain.coords_index
        else:
            out_namedTable = self.__class__(df)
            return out_namedTable

    def rename(self, domain_names_map=None, columns=None, inplace=False):
        """
        Rename either domain names or data columns of this namedTable.

        It is not an error if the mapping(s) contain keys that are not in the
        corresponding domain names/columns. This mirrors pandas DataFrame rename
        behavior. (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html)

        Args:
            domain_names_map (dict): Keys are original names of this namedTable's domain,
                values are new names.
            columns (dict): Keys are original names, values are new names.
            inplace (bool): If True, modify the namedTable inplace, else return a
                new namedTable.

        Returns:
            DFBasenamedTable|None: If inplace=True, returns a new namedTable. Otherwise
                modifies this namedTable's internal dataframe.
        """
        domain_names_map = domain_names_map or {}
        columns = columns or {}

        if not isinstance(domain_names_map, dict) or not isinstance(columns, dict):
            raise ValueError('domain_names_map and/or columns must be dict type.')

        if not domain_names_map and not columns:
            logger.warning('Both domain_names_map and columns are None. rename()'
                           ' will be a no-op.')

        if inplace:
            df = self._df
        else:
            df = self._df.copy()

        if domain_names_map:
            new_names = []

            for name in self.domain.names:
                try:
                    new_name = domain_names_map[name]
                except KeyError:
                    new_name = name

                new_names.append(new_name)

            assert isinstance(df.index, pd.MultiIndex), 'Underlying df index should be a MultiIndex.'
            df.index.names = new_names

        if columns:
            df.rename(columns=columns, inplace=True)

        if inplace:
            self._df = df
            self.domain.names = df.index.names
        else:
            return self.__class__(df)

    def mask_reindex(self, mask, inplace=True):
        """
        Reindex based on mask
        Args:
            mask (array bool):
            inplace (bool):
        """
        if len(mask) != len(self.df) or mask.ndim != 1:
            raise ValueError("expects one dimensional vector same length as namedTable's df")
        if inplace:
            self._df = self._df[mask]
            self.domain = self.domain.get_mask_sub_domain(mask)
            self._df.index = self.domain.coords_index
        else:
            df = self.df[mask]
            df.index = self.domain.get_mask_sub_domain(mask).coords_index
            return self.__class__(df)

    def transform(self,
                  func,
                  target_cols=None,
                  inplace=False,
                  *args,
                  **kwargs):
        """
        Transform the specified data columns in the namedTable

        Args:
            func: the transformation function
            target_cols: the names of the columns to be transformed, all data columns if None
            inplace: modify the namedTable inplace if True, else return a new namedTable
            *args: non-keyword arguments for func
            **kwargs: keyword arguments for func

        Returns:
            None if inplace == True,
            the transformed DFBasenamedTable if inplace == False
        """

        if target_cols is None:
            target_cols = self.data_column_names
        tmp_df = self.df[target_cols].transform(func,
                                                *args,
                                                **kwargs)
        if inplace:
            self._df[target_cols] = tmp_df.copy()
            return None
        else:
            return self.__class__(tmp_df)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.domain == other.domain and self.df.equals(other.df) \
               and (self.data_column_names == other.data_column_names).all()

    # TODO: Make binary operations more efficient by replacing df.loc() with faster operations
    def __add__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] + other.df.loc[idx_intersect])

    def __radd__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] + other.df.loc[idx_intersect])

    def __sub__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] - other.df.loc[idx_intersect])

    def __rsub__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] - other.df.loc[idx_intersect])

    def __mul__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] * other.df.loc[idx_intersect])

    def __rmul__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] * other.df.loc[idx_intersect])

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] / other.df.loc[idx_intersect])

    def __rdiv__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] / other.df.loc[idx_intersect])

    def __pow__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] ** other.df.loc[idx_intersect])

    def __rpow__(self, other):
        if not isinstance(other, DFBasenamedTable):
            raise ValueError('operand must be DFBasenamedTable Object')

        idx_intersect = self.domain.coords_index.intersection(other.domain.coords_index)
        if len(idx_intersect) == 0:
            raise ValueError('the operands do not share any indices')
        return self.__class__(self.df.loc[idx_intersect] ** other.df.loc[idx_intersect])

    @staticmethod
    def _append(first, second, output_class):
        if first.domain.names != second.domain.names:
            raise ValueError('Append only available for namedTables with domains that have the same coordinate names')
        if first.data_column_names != second.data_column_names:
            raise ValueError('Append only available for namedTables with the same data column names')

        return output_class(pd.concat([first.df, second.df], axis=0, sort=True))

    def append(self, other):
        self._append(self, other, self.__class__)

    def _groupby(self, by, agg_func='sum'):

        if isinstance(by, (str,)):
            by = [by]

        if not set(by) <= set(self.domain.names):
            raise ValueError('by = %s is (are) not a coordinate(s) in domain.' % by)
        level = None if type(self.df.index) == pd.Index else by

        g = self.df.groupby(axis=0, level=level,
                            as_index=True, sort=False)
        return g.agg(agg_func)

    def groupby(self, by, agg_func='sum'):
        """
        Groups the namedTable by a coordinate(s) of the domain and applies agg_func to create a new namedTable.
        Args:
            by: coordinates in domain to group by for
            agg_func: function to use for aggregation (should be compatible to pandas groupby agrgegation function)

        """

        df = self._groupby(by, agg_func)

        return self.__class__(df)

    def filter(self, func, *args, **kwargs):
        return self.__class__(self.df[self.transform(func, *args, **kwargs).df].dropna())

    def clip(self, lower=None, upper=None, by=None, target_col=None, inplace=False, quantile=False, clip_behavior='remove'):
        """
        Clip the values contained in the namedTable.

        Args:
            lower: minimum value or lower quantile
            upper: maximum value or upper quantile
            by: the list of indices to group by.
            target_col: name or list of data column names to clip. If None,
                performs clipping on all data columns. Names may be specified
                in any order but will not change the namedTable data column order.
                If more than one name is provided and `clip_behavior='remove'`,
                only the rows for which all their values are conforming are
                retained.
            inplace: modify namedTable in place or return new namedTable
            quantile: if True, treat lower and upper as quantiles
            clip_behavior: Describes how treat values outside acceptable range.
                Available options:
                "remove": default, remove values. If `by` is not None, find the
                max values for each group then remove groups whose max value is
                not within the lower and upper (quantiles if quantile=True)
                of all the group max values.
                "bound": clip non conforming values to the specified bounds.
                Raises ValueError if `by` is not None.

        Returns:
            None if inplace is True
            a new namedTable of the same type if inplace is False
        """
        _ALLOWED = {'remove', 'bound'}
        lower = lower if lower is not None else 0. if quantile else -np.inf
        upper = upper if upper is not None else 1. if quantile else np.inf
        assert lower < upper, 'lower bound/quantile must be less than upper bound/quantile'
        assert clip_behavior in _ALLOWED, 'invalid clip_behavior specified {}'.format(clip_behavior)

        if target_col is None:
            target_col = self.data_column_names
        elif isinstance(target_col, (str,)):
            target_col = [target_col]
        else:
            # TODO: do we want to allow user to control the output column order
            # in the namedTable? The below code does NOT allow user the control.
            # ensure user specified columns are sorted in same relative order
            # as self.data_column_names
            target_col.sort(key=lambda x: self.data_column_names.get_loc(x))

        assert set(target_col).issubset(self.data_column_names), \
            'Unknown data column names provided: {}'.format(np.setdiff1d(target_col, self.data_column_names))

        df = self.df.copy() if not inplace else self.df
        if by is None:
            lower = self.df[target_col].quantile(lower) if quantile else lower
            upper = self.df[target_col].quantile(upper) if quantile else upper
            if clip_behavior == 'remove':
                mask = np.asarray((lower <= df[target_col]) & (df[target_col] <= upper))
                if len(target_col) > 1:
                    # Only retain rows where all values in mask are True.
                    mask = mask.all(axis=1)
            elif clip_behavior == 'bound':
                # If inplace=False, the copy is modified in-place
                df[target_col] = df[target_col].clip(lower=lower, upper=upper, axis=1)
                if inplace:
                    self._df = df
                    return
                return self.__class__(df)
            else:
                raise NotImplementedError
        else:
            if clip_behavior == 'remove':
                max_group_vals = df.groupby(by=by)[target_col].transform('max')
                # if quantile=True, and more than one target_col, consider
                # quantiles on per-column basis
                # NOTE: may be scalar or Series
                lower = max_group_vals.quantile(lower) if quantile else lower
                upper = max_group_vals.quantile(upper) if quantile else upper

                mask = np.asarray((lower <= max_group_vals) & (max_group_vals <= upper))
                if len(target_col) > 1:
                    # Only retain rows where all values in mask are True.
                    mask = mask.all(axis=1)
            elif clip_behavior == 'bound':
                # TODO: bounded clip behavior undefined at group level
                raise ValueError('`bound` option is invalid when `by` is not None')
            else:
                raise NotImplementedError
        return self.mask_reindex(mask.ravel(), inplace=inplace)

    def expand(self, domain, fill_with):
        """
        Adds to the namedTable all elements in domain not present in self.domain with value equal to fill_with.
        Domain comparison
        Args:
            domain (DFDomain):
            fill_with (numerical):

        Returns:
            a new namedTable with expanded domain
        """
        if self.domain.names != domain.names:
            raise ValueError('DFBaseDomain.expand only available for namedTables with domains that have the same '
                             'coordinate names')

        intersection_mask = domain.get_intersection_mask(self.domain)
        append_domain = domain.get_mask_sub_domain(~intersection_mask)

        def get_new_df(index):
            df_mat = self.df.to_coo() if isinstance(self.df, pd.SparseDataFrame) else self.df.values
            fill_mask = np.full(df_mat.shape[0] + len(append_domain) ,True)
            fill_mask[df_mat.shape[0]:] = False
            new_df_mat = merge_utils.mat_add_rows_with_value(df_mat, fill_mask, fill_with)
            if isinstance(self.df, pd.SparseDataFrame):
                if fill_with != self.df.default_fill_value:
                    logger.warning("Filling sparse dataframe with non-default_fill_value. This will make it dense. "
                                   "default_fill_value: {}, fill_na: {}.".format(self.df.default_fill_value,
                                                                                 fill_with))

                new_df = pd_compat.SparseDataFrame(new_df_mat, index=index,
                                                   columns=self.data_column_names)
            else:
                new_df = pd.DataFrame(new_df_mat,index=index, columns=self.data_column_names)

            return new_df

        appended_index = self.domain.append(append_domain).coords_index
        try:  # try to cast append_domain to type of self.domain
            appended_domain = type(self.domain)(appended_index)
            return type(self)(get_new_df(appended_domain.coords_index))
        except DFDomainCreationError:
            raise ValueError('Input domain is not of the same type as the domain of '
                             'the namedTable and could not be cast.')


