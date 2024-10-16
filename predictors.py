from typing import Callable, Dict, Iterator, NamedTuple, Optional

import numpy as np
import pandas as pd
import toolz

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor

try:
    # from sktime.forecasting.statsforecast import StatsForecastAutoARIMA as autoARIMA
    from sktime.forecasting.arima import AutoARIMA as autoARIMA
except ImportError:
    autoARIMA = None
from gluonts.ext.prophet import ProphetPredictor as ProphetPredictor
from gluonts.model.seasonal_naive import (
    SeasonalNaivePredictor as SeasonalNaivePredictor,
)


SKT_AUTOARIMA_IS_INSTALLED = autoARIMA is not None

USAGE_MESSAGE = """
Cannot import `sktime.forecasting.statsforecast.StatsForecastAutoARIMA`.

The `ARIMAPredictor` is a thin wrapper for calling the `pmdarima.arima.AutoARIMA` package.
In order to use it you need to install it using the following two
methods:

    # 1) install directly
    pip install pmdarima sktime

"""


class ARIMADataEntry(NamedTuple):
    """
    A named tuple containing relevant base and derived data that is required in
    order to call AutoARIMA.
    """

    train_length: int
    prediction_length: int
    start: pd.Period
    target: np.ndarray

    @property
    def arima_training_data(self) -> pd.Series:
        return pd.DataFrame(
            data={
                **{
                    "ds": pd.period_range(
                        start=self.start,
                        periods=self.train_length,
                        freq=self.start.freq,
                    ).to_timestamp(),
                    "y": self.target,
                },
            }
        ).set_index("ds")

    @property
    def forecast_start(self) -> pd.Period:
        return self.start + self.train_length * self.start.freq

    @property
    def freq(self):
        return self.start.freq


class ARIMAPredictor(RepresentablePredictor):
    """
    Wrapper around `sktime.forecasting.arima.AutoARIMA` to expose it to gluonts.

    The `ARIMAPredictor` is a thin wrapper for calling the `pmdarima.arima.AutoARIMA` package.
    In order to use it you need to install it using the following two
    methods:

        # 1) install directly
        pip install pmdarima sktime

    Parameters
    ----------
    prediction_length
        Number of time points to predict
    arima_params
        Parameters to pass when instantiating the autoARIMA model.
    init_model
        An optional function that will be called with the configured model.
        This can be used to configure more complex setups, e.g.

        >>> def configure_model(model):
        ...     model.add_seasonality(
        ...         name='weekly', period=7, fourier_order=3, prior_scale=0.1
        ...     )
        ...     return model
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        arima_params: Optional[Dict] = None,
        init_model: Callable = toolz.identity,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        if not SKT_AUTOARIMA_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        if arima_params is None:
            arima_params = {}
        arima_params.setdefault("seasonal", True)
        arima_params.setdefault("sp", season_length)
        arima_params.setdefault("max_order", 10)
        arima_params.setdefault("maxiter", 500)
        arima_params.setdefault("suppress_warnings", True)

        self.arima_params = arima_params
        self.init_model = init_model

    def predict(
        self, dataset: Dataset, num_samples: int = 100, **kwargs
    ) -> Iterator[SampleForecast]:
        params = self.arima_params.copy()

        for entry in dataset:
            data = self._make_ARIMA_data_entry(entry)
            try:
                forecast_samples = self._run_ARIMA(data, params, num_samples)
            except ValueError as _:
                # OJO - "seasonal" should not be set to false to give good results
                # this is a workaround for the error
                # `ValueError: shapes (4,2) and (1,) not aligned: 2 (dim 1) != 1 (dim 0)`
                # when running against 2:1 ratio on week10
                print("WARN: ARIMA failed, retrying with seasonal=False")
                params["seasonal"] = False
                forecast_samples = self._run_ARIMA(data, params, num_samples)

            yield SampleForecast(
                samples=forecast_samples,
                start_date=data.forecast_start,
                item_id=entry.get("item_id"),
                info=entry.get("info"),
            )

    def _run_ARIMA(self, data: ARIMADataEntry, params: dict, num_samples) -> np.ndarray:
        """
        Construct and run a :class:`ARIMA` model on the given
        :class:`ARIMADataEntry` and return the resulting array of samples.
        """
        forecaster = self.init_model(autoARIMA(**params))
        forecaster.fit_predict(
            y=data.arima_training_data, fh=np.arange(data.prediction_length)
        )

        # An attempt was made to generate confidence intervals by predicting quantiles

        quantiles = np.linspace(0.01, 1, num_samples, endpoint=False)
        forecast_ci = forecaster.predict_quantiles(
            fh=np.arange(data.prediction_length), alpha=quantiles
        )

        return forecast_ci.T.to_numpy(dtype=np.float64)

    def _make_ARIMA_data_entry(self, entry: DataEntry) -> ARIMADataEntry:
        """
        Construct a :class:`ARIMADataEntry` from a regular
        :class:`DataEntry`.
        """

        train_length = len(entry["target"])
        prediction_length = self.prediction_length
        start = entry["start"]
        target = entry["target"]

        return ARIMADataEntry(
            train_length=train_length,
            prediction_length=prediction_length,
            start=start,
            target=target,
        )
