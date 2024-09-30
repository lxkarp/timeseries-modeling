from gluonts.ev.metrics import (
    BaseMetricDefinition,
    DirectMetric,
    DerivedMetric,
    WeightedSumQuantileLoss,
    MASE,
    MeanWeightedSumQuantileLoss,
)

from gluonts.ev.aggregations import Aggregation
from scipy.stats import wasserstein_distance
from gluonts.ev.aggregations import Mean
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.stats import absolute_scaled_error

from functools import partial
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import (
    Collection,
    Optional,
    Dict,
    List,
)


def wd(data, forecast_type: str) -> np.ndarray:
    return_wd: List[np.float64] = []

    for i in range(len(data["label"])):
        return_wd.append(
            wasserstein_distance(data["label"][i]._get_data(), data[forecast_type][i])
        )
    return np.array(return_wd)


def swd(data, forecast_type: str) -> np.ndarray:
    """
    scaled wasserstein distance
    """
    return_wd: List[np.float64] = []

    for i in range(len(data["label"])):
        norm_pred = data["label"][i]._get_data() / data["seasonal_error"][i]
        norm_actuals = data[forecast_type][i] / data["seasonal_error"][i]
        return_wd.append(wasserstein_distance(norm_pred, norm_actuals))
    return np.array(return_wd)


@dataclass
class EMD(BaseMetricDefinition):
    """
    Earth Mover's Distance (EMD) metric.
    """
    q: float

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: int) -> DirectMetric:
        return DirectMetric(
            name=f"EMD[{self.q}]",
            stat=partial(swd, forecast_type=self.q),
            aggregate=Mean(axis=0),
        )

@dataclass
class MeanDecileEMD(BaseMetricDefinition):
    """
    Mean of Earth Mover's Distance (EMD) metric across deciles.
    """
    quantile_levels = np.arange(0.1, 1.0, 0.1)

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        stacked_quantile_losses = np.stack(
            [quantile_loss for quantile_loss in quantile_losses.values()],
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: int) -> DirectMetric:
        return DerivedMetric(
            name="MeanDecileEMD",
            metrics={
                f"EMD[{q}]": EMD(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.mean,
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
            stat=partial(absolute_scaled_error, forecast_type=self.forecast_type),
            aggregate=ListAgg(axis=axis),
        )


@dataclass
class MeanWeightedSumQuantileLossna(BaseMetricDefinition):
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
                f"quantile_loss[{q}]": WeightedSumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.noagg,
        )


@dataclass
class ListAgg(Aggregation):
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
            print()
            print("values: ", values)
            print("partial result: ", self.partial_result)
            self.partial_result = np.concatenate([self.partial_result, values])
        else:
            assert isinstance(self.partial_result, np.ndarray)
            self.partial_result = np.concatenate([self.partial_result, values])

    def get(self) -> np.ndarray:
        assert self.axis is None or isinstance(self.axis, tuple)

        if self.axis is None or 0 in self.axis:
            return self.partial_result

        assert isinstance(self.partial_result, list)
        return np.concatenate(self.partial_result)


def mk_metrics(context, forecast):
    metrics = (
        evaluate_forecasts(
            forecast,
            test_data=context,
            metrics=[
                MASE(),
                MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                MeanDecileEMD(),
            ],
            batch_size=5000,
        )
        .reset_index(drop=True)
        .rename(
            {
                "MASE[0.5]": "MASE",
                "mean_weighted_sum_quantile_loss": "WQL",
                "MeanDecileEMD": "EMD",
            },
            axis="columns",
        )
        .to_dict(orient="records")
    )

    return metrics[0]  # !!OJO!! Magic numbers to remove the list


def mk_viz(context, forecast, config):
    metrics = mk_metrics(context, forecast)
    _context = context.label

    forecasts = forecast[0]
    cat = config["category"]
    ratio = config['prediction_ratio']

    graph_data_length = len(_context.test_data.dataset[0]["target"])

    context_data_start = _context.test_data.dataset[0]["start"].to_timestamp()

    plot_dates = pd.date_range(
        start=context_data_start,
        periods=graph_data_length,
        freq=_context.test_data.dataset[0]["start"].freq,
    )
    fig, ax = plt.subplots()

    # plot the line of all the actuals

    ax.plot(plot_dates, _context.test_data.dataset[0]["target"])

    forecasts.plot(ax=ax, show_label=True)
    fig.autofmt_xdate()
    plt.suptitle(
        f'{config["model_name"]} {ratio} {cat} {config["segment_name"]}', fontsize=18
    )
    plt.title(
        "metrics: EMD:{EMD:.4f}, MASE:{MASE:.4f}, WQL:{WQL:.4f}".format(**metrics),
        fontsize=10,
        y=1,
    )
    plt.legend()
    plt.savefig(f'./{config["model_name"]}_{ratio}_{cat}_{config["segment_name"]}.png')
