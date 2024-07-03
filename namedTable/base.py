# /usr/bin/python
# coding=utf-8
"""Base class
"""
import numpy as np

from namedTable.domains.base import Domain
from namedTable.domains.exceptions import DomainKeyError


class namedTable(object):
    """
    # TODO: could we have a MultinamedTable?

    This represents an index that will be used as an input to a regressor.

    (Note that the target in standard Nabee usage should also be of this type.)
    """

    def __init__(self, domain, **kwargs):
        """
        Initialize / build the namedTable index.

        Args:
            domain (namedTable.domains.base.Domain): domain over which namedTable is defined
            kwargs (dict): used by subclasses
        """
        if not isinstance(domain, Domain):
            raise ValueError("domain must be an instance of Domain.")

        self.domain = domain

    def __getitem__(self, item):
        """
        This should only raise an error if `item not in self.domain`. Otherwise, this should always
        return something.

        Args:
            item (object that the domain understands)

        Returns:
            float or None

        Raises:
            DomainKeyError when item not in self.domain
        """
        if item not in self.domain:
            raise DomainKeyError("{} not found in domain.".format(item))
        else:
            return self._get(item)

    def _get(self, item):
        """
        Implementation of actual getter.

        Args:
            item (object that the domain understands): can assume this is in the domain

        Returns:
            float or None
        """
        raise NotImplementedError

    def ravel(self):
        """
        Returns:
            x (np.ndarray): index raveled over given domain. Order is expected to match domain ravel order
        """
        # NOTE: this base implementation may be inefficient. This should be overridden in subclasses for efficiency.
        return np.asarray([self[item] for item in self.domain])

    @classmethod
    def unravel(cls, domain, data, **kwargs):
        """
        Args:
            domain (instance of Domain): domain to create namedTable. Should be aligned with data.
            data (numpy array): The data of the namedTable. Should be of length equal to the size of the domain.

        Returns:
            a namedTable with the domain and data passed as arguments
        """
        raise NotImplementedError

    def format_as_time_series_df(self, **kwargs):
        """
        Returns the namedTable as a pandas DataFrame formatted as a (multivalued) time series.
        The index of the dataframe is the time periods (as datetime objects);
        the columns are every (row,col) combination.

        If not implemented, cannot use the namedTable subclass with multi_fit_and_forecast in nabee.

        Returns:
            df (pd.DataFrame)
        """
        raise NotImplementedError


class namedTableFactory(object):

    @staticmethod
    def create(**kwargs):
        raise NotImplementedError()
