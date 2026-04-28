from gluonts.ev.metrics import (
    BaseMetricDefinition,
    DirectMetric,
    DerivedMetric,
    WeightedSumQuantileLoss,
    MASE,
    MeanWeightedSumQuantileLoss,
    NRMSE,
    SMAPE,
)

from gluonts.ev.aggregations import Aggregation
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.ev.aggregations import Mean
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.stats import absolute_scaled_error

from functools import partial
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from typing import (
    Collection,
    Optional,
    Dict,
    List,
)

from pemd import (
    probabilistic_value_distribution_emd,
    value_distribution_emd,
)


def _label_values(label) -> np.ndarray:
    if hasattr(label, "_get_data"):
        label = label._get_data()
    return np.asarray(label, dtype=float).reshape(-1)


def _data_at(data, key, index: int) -> np.ndarray:
    try:
        values = data[key][index]
    except KeyError:
        values = data[str(key)][index]
    return np.asarray(values, dtype=float).reshape(-1)


def _seasonal_scale(data, index: int) -> float:
    scale = np.asarray(data["seasonal_error"][index], dtype=float)
    return float(np.nanmean(scale))


def _scaled(values: np.ndarray, scale: Optional[float]) -> np.ndarray:
    if scale is None:
        return values
    return values / scale


def _forecast_samples(
    data,
    index: int,
    quantile_levels: Optional[Collection[float]] = None,
) -> np.ndarray:
    forecast = data.maps[1].forecasts[index]

    if isinstance(forecast, SampleForecast) or hasattr(forecast, "samples"):
        samples = np.asarray(forecast.samples, dtype=float)
    elif isinstance(forecast, QuantileForecast) or quantile_levels is not None:
        if quantile_levels is None:
            raise ValueError("quantile_levels are required for QuantileForecast pEMD")
        samples = np.asarray([_data_at(data, q, index) for q in quantile_levels])
    else:
        raise TypeError(
            "pEMD requires SampleForecast samples or quantile levels for "
            "a QuantileForecast fallback"
        )

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError("forecast samples must have shape (num_samples, horizon)")
    return samples


def emd_stat(
    data,
    forecast_type: str = "0.5",
    scaled: bool = True,
) -> np.ndarray:
    """
    Value-distribution EMD for each item in an evaluation batch.

    This is Wasserstein-1 between empirical distributions of horizon values.
    When ``scaled`` is true, both horizons are divided by the MASE seasonal
    scale before computing EMD, yielding the scaled variant used in tables.
    """
    distances: List[float] = []
    for i in range(len(data["label"])):
        scale = _seasonal_scale(data, i) if scaled else None
        if scaled and (not np.isfinite(scale) or scale == 0.0):
            distances.append(np.nan)
            continue

        actuals = _scaled(_label_values(data["label"][i]), scale)
        forecast = _scaled(_data_at(data, forecast_type, i), scale)
        distances.append(value_distribution_emd(actuals, forecast))
    return np.array(distances)[:, np.newaxis]


def pemd_stat(
    data,
    quantile_levels: Optional[Collection[float]] = None,
    scaled: bool = True,
) -> np.ndarray:
    """
    Probabilistic value-distribution EMD for each item in a batch.

    For SampleForecasts this uses forecast sample trajectories directly. For
    QuantileForecasts, quantile curves are used as a fallback ensemble via the
    comonotone coupling implied by shared quantile levels across the horizon.
    """
    distances: List[float] = []
    for i in range(len(data["label"])):
        scale = _seasonal_scale(data, i) if scaled else None
        if scaled and (not np.isfinite(scale) or scale == 0.0):
            distances.append(np.nan)
            continue

        actuals = _scaled(_label_values(data["label"][i]), scale)
        samples = _scaled(_forecast_samples(data, i, quantile_levels), scale)
        distances.append(probabilistic_value_distribution_emd(actuals, samples))
    return np.array(distances)[:, np.newaxis]


def mk_spread(timeseries, num_samples: int, delta: bool = True) -> np.ndarray:
    if not delta:
        return np.tile(timeseries, (num_samples, 1))
    interim = np.zeros(shape=(num_samples, len(timeseries)))
    interim[num_samples // 2] = timeseries
    return interim


@dataclass
class EMD(BaseMetricDefinition):
    """
    Value-distribution Earth Mover's Distance (EMD) metric.
    """
    forecast_type: str = "0.5"
    scaled: bool = True

    def __call__(self, axis: int) -> DirectMetric:
        return DirectMetric(
            name="EMD",
            stat=partial(
                emd_stat,
                forecast_type=self.forecast_type,
                scaled=self.scaled,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class PEMD(BaseMetricDefinition):
    """
    Probabilistic value-distribution EMD for sampled forecast trajectories.
    """
    quantile_levels: Optional[Collection[float]] = None
    scaled: bool = True

    def __call__(self, axis: int) -> DirectMetric:
        return DirectMetric(
            name="pEMD",
            stat=partial(
                pemd_stat,
                quantile_levels=self.quantile_levels,
                scaled=self.scaled,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MeanDecileEMD(BaseMetricDefinition):
    """
    Mean of Earth Mover's Distance (EMD) metric across deciles.
    """

    quantile_levels: Collection[float]

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
                f"EMD[{q}]": EMD(forecast_type=str(q))(axis=axis)
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
                # MeanDecileEMD(np.arange(0.1, 1.0, 0.1)),
                EMD(),
                PEMD(np.arange(0.1, 1.0, 0.1)),
                NRMSE(),
                SMAPE(),
            ],
            batch_size=5000,
        )
        .reset_index(drop=True)
        .rename(
            {
                "MASE[0.5]": "MASE",
                "mean_weighted_sum_quantile_loss": "WQL",
                # "MeanDecileEMD": "mdEMD",
                "EMD": "EMD",
                "pEMD": "pEMD",
                # "NRMSE[mean]": "NRMSE",
                # "sMAPE[0.5]": "SMAPE",
            },
            axis="columns",
        )
        .to_dict(orient="records")
    )
    print(metrics)

    return metrics[0]  # !!OJO!! Magic numbers to remove the list


def save_metrics_to_csv(metrics, config, output_path):
    df = pd.DataFrame(
        [
            {
                "model_name": config["model_name"],
                "ratio": config["prediction_ratio"],
                "category": config["category"],
                "segment_name": config["segment_name"],
                # "mdEMD": metrics["mdEMD"],
                "MASE": metrics["MASE"],
                "WQL": metrics["WQL"],
                "EMD": metrics["EMD"],
                "pEMD": metrics["pEMD"],
                # "NRMSE": metrics["NRMSE"],
                # "SMAPE": metrics["SMAPE"],
            }
        ]
    )

    try:
        if not os.path.exists("./out"):
            os.makedirs("./out")

        if os.path.exists(output_path):
            df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            df.to_csv(output_path, mode="w", header=True, index=False)
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")


def mk_viz(context, forecast, config):
    metrics = mk_metrics(context, forecast)
    _context = context.label

    forecasts = forecast[0]
    cat = config["category"]
    ratio = config["prediction_ratio"]

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
        (
            "metrics: EMD:{EMD:.4f}, pEMD:{pEMD:.4f}, "
            "MASE:{MASE:.4f}, WQL:{WQL:.4f}"
        ).format(**metrics),
        fontsize=10,
        y=1,
    )

    plt.legend()
    plot_dir = os.environ.get("PLOT_DIR", "./out")
    os.makedirs(plot_dir, exist_ok=True)
    safe_ratio = str(ratio).replace(":", "-")
    plt.savefig(
        os.path.join(
            plot_dir,
            f'{config["model_name"]}_{safe_ratio}_{cat}_{config["segment_name"]}.png',
        )
    )
    return metrics
