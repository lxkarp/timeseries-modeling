from evaluate import generate_sample_forecasts, ChronosPipeline, load_and_split_dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, BaseMetricDefinition, DirectMetric
from scipy.stats import wasserstein_distance
from gluonts.ev.aggregations import Mean

import os
import yaml
from functools import partial
from dataclasses import dataclass
from pprint import pprint as print

from typing import (
    Collection,
    Optional,
    Callable,
    Mapping,
    Dict,
    List,
    Iterator,
)
from typing_extensions import Protocol, runtime_checkable, Self


# config
chronos_model_id  = "amazon/chronos-t5-small"
device = "cuda:0"
torch_dtype = "bfloat16"
batch_size = 32
num_samples = 20

pipeline = ChronosPipeline.from_pretrained(
    chronos_model_id,
    device_map=device,
    torch_dtype=torch_dtype,
)

def load_config(path : str):
    if not config_file_path:
        raise ValueError("You must set the environment variable $CHRONOS_EVAL_CONFIG")
    with open(config_file_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    # !!! you can only have a single config in your yaml
    config = backtest_configs[0] 
    return config

def mk_forecasts(test_data):
    sample_forecasts = generate_sample_forecasts(
        test_data.input,
        pipeline=pipeline,
        prediction_length=config["prediction_length"],
        batch_size=batch_size,
        num_samples=num_samples,
        temperature=None,
        top_k=None,
        top_p=None,
    )
    return sample_forecasts

def mk_metrics(context, forecast):
    metrics = (
            evaluate_forecasts(
                forecast,
                test_data=context,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

    return metrics

def mk_viz(context, forecast, config):
    context_by_category = {'casual': context[0][0], 'registered': context[1][0]}
    actuals_by_category = {'casual': context[0][1], 'registered': context[1][1]}
    forecasts_by_category = {'casual': forecast[0], 'registered': forecast[1]}

    # registered only atm
    context_data_length = len(context_by_category['registered']['target'])
    context_data_start = context_by_category['registered']['start'].to_timestamp()

    plot_dates = pd.date_range(start=context_data_start, periods=config['prediction_length']+context_data_length, freq='D')
    fig, ax = plt.subplots()

    ax.plot(plot_dates, np.append(context_by_category['registered']['target'], actuals_by_category['registered']['target']))
    forecasts_by_category['registered'].plot(ax=ax, show_label=True)
    fig.autofmt_xdate()
    plt.title('Registered July Forecasts')
    plt.legend()
    plt.savefig('./test_viz_png.png')


def wd(data ,forecast_type: str) -> float:
    return wasserstein_distance(data['label'], data[forecast_type])

@dataclass 
class EMD(BaseMetricDefinition):
    """
    Earth Mover's Distance (EMD) metric.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"EMD_{self.forecast_type}",
            stat=partial(wd, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis), 
        )
    


if __name__ == '__main__':
    emd = EMD()
    A = [0,1, 3] 
    B = [5, 6, 8]
    Data = {'label': A, 'forecast': B}
    forecast_type = 'forecast'
    print(wd(Data, forecast_type))
    
    
    if False:
        config_file_path = os.environ.get('CHRONOS_EVAL_CONFIG')
        config = load_config(config_file_path)
        test_data = load_and_split_dataset(backtest_config=config)
        forecast = mk_forecasts(test_data)
        mk_viz([item for item in test_data], forecast, config)
        print(mk_metrics(test_data, forecast))
