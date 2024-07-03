# /usr/bin/python
# coding=utf-8
"""Constant-valued namedTable
"""
import numpy as np

from namedTable.base import namedTable


class ConstantnamedTable(namedTable):
    def __init__(self, domain, value):
        super(ConstantnamedTable, self).__init__(domain)

        self.value = value

    def _get(self, item):
        return self.value

    def ravel(self):
        return np.full(len(self.domain), self.value)
