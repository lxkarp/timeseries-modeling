from gluonts.ev.metrics import (
    BaseMetricDefinition,
    DirectMetric,
    DerivedMetric,
    SumQuantileLoss,
)
from gluonts.ev.aggregations import Aggregation
from scipy.stats import wasserstein_distance
from gluonts.ev.aggregations import Mean

from gluonts.ev.stats import absolute_scaled_error

from functools import partial
from dataclasses import dataclass

import numpy as np

from typing import (
    Collection,
    Optional,
    Dict,
    List,
)


def wd(data, forecast_type: str) -> np.ndarray:
    return_wd: List[np.float64] = []

    for i in range(len(data['label'])):
        return_wd.append(wasserstein_distance(data['label'][i]._get_data(), data[forecast_type][i]))
    return np.array(return_wd)


def swd(data, forecast_type: str) -> np.ndarray:
    """
    scaled wasserstein distance
    """
    return_wd: List[np.float64] = []

    for i in range(len(data['label'])):
        norm_pred = data['label'][i]._get_data() / data['seasonal_error'][i]
        norm_actuals = data[forecast_type][i] / data['seasonal_error'][i]
        return_wd.append(wasserstein_distance(norm_pred, norm_actuals))
    return np.array(return_wd)


@dataclass 
class EMDna(BaseMetricDefinition):
    """
    Earth Mover's Distance (EMD) metric.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: int) -> DirectMetric:
        return DirectMetric(
            name=f"EMD[{self.forecast_type}]",
            stat=partial(swd, forecast_type=self.forecast_type),
            aggregate=ListAgg(axis=0),  # hard code as kludge
        )


@dataclass
class MASEna(BaseMetricDefinition):
    """
    Mean Absolute Scaled Error.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"MASE[{self.forecast_type}]",
            stat=partial(
                absolute_scaled_error, forecast_type=self.forecast_type
            ),
            aggregate=ListAgg(axis=axis),
        )


@dataclass
class MeanSumQuantileLossna(BaseMetricDefinition):
    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    @staticmethod
    def noagg(**quantile_losses: np.ndarray) -> Dict[str, np.ndarray]:
        return np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name="mean_sum_quantile_loss",
            metrics={
                f"quantile_loss[{q}]": SumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.noagg,
        )


@dataclass
class ListAgg(Aggregation[np.ndarray]):
    """
    Map-reduce way of collecting values into a list.

    `partial_result` represents one of two things, depending on the axis:
    Case 1 - axis 0 is aggregated (axis is None or 0):
        In each `step`, values are being collected into `partial_result` list.

    Case 2 - axis 0 is not being aggregated:
        In this case, `partial_result` is a list that in the end gets
        concatenated to a np.ndarray.
    """

    partial_result: Optional[List[np.ndarray]] = None

    def step(self, values: np.ndarray) -> None:
        assert self.axis is None or isinstance(self.axis, tuple)

        if self.partial_result is None:
            self.partial_result = []

        if self.axis is None or 0 in self.axis:
            self.partial_result.append(values)
        else:
            assert isinstance(self.partial_result, list)
            self.partial_result.append(values)

    def get(self) -> np.ndarray:
        assert self.axis is None or isinstance(self.axis, tuple)

        if self.axis is None or 0 in self.axis:
            return self.partial_result

        assert isinstance(self.partial_result, list)
        return np.concatenate(self.partial_result)
