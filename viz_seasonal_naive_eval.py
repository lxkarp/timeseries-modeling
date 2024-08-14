from chronos_utils import load_and_split_dataset
from metrics import EMD
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.evaluation import make_evaluation_predictions

import os
import yaml
# from pprint import pprint as print

model_name = "Naive"

def load_config(config_file_path : str):
    with open(config_file_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    # !!! you can only have a single config in your yaml
    config = backtest_configs[0]
    return config


def mk_forecasts(test_data, config):
    prediction_length = np.abs(config['offset'])
    season_length = config['expected_seasonality']
    pipeline = SeasonalNaivePredictor(prediction_length=prediction_length, season_length=season_length)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=pipeline,  # predictor
        num_samples=1,  # number of sample paths we want for evaluation
    )

    sample_forecasts = list(forecast_it)

    return sample_forecasts


def mk_metrics(context, forecast):
    metrics = evaluate_forecasts(
                forecast,
                test_data=context,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                    EMD(),
                ],
                batch_size=5000,
    ).reset_index(
            drop=True
    ).rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL", "EMD[0.5]":"EMD"},
            axis="columns",
    ).to_dict(orient="records")

    return metrics[0]  # !!OJO!! Magic numbers to remove the list


def mk_viz(context, forecast, config):
    metrics = mk_metrics(context, forecast)
    # _context = [item for item in context]
    _context = context

    # just the training data, no data for any of the prediction period
    # context = _context[0]
    # just the ground-truth
    # actuals = _context[1]
    forecasts = forecast[0]

    # for cat, cat_data in context_by_category.items():
    cat = config['category']

    print(_context.input)
    context_data_length = len(_context.input['target'])
    context_data_start = _context.input['start'].to_timestamp()

    plot_dates = pd.date_range(start=context_data_start, periods=config['prediction_length']+context_data_length, freq=_context.input['start'].freq)
    fig, ax = plt.subplots()

    ax.plot(plot_dates, _context.input)
    # ax.plot(plot_dates, np.append(cat_data['target'], actuals_by_category[cat]['target']))
    forecasts.plot(ax=ax, show_label=True)
    fig.autofmt_xdate()
    plt.suptitle(f'{model_name} {cat} {config["segment_name"]} Forecasts', fontsize=18)
    plt.title('metrics: EMD:{EMD}, MASE:{MASE}, WQL:{WQL}'.format(**metrics), fontsize=10, y=1)
    plt.legend()
    plt.savefig(f'./{model_name}_{cat}_{config["segment_name"]}.png')


if __name__ == '__main__':
    config_file_path = os.environ.get('CHRONOS_EVAL_CONFIG')
    if config_file_path is None:
        raise ValueError("You must set the environment variable $CHRONOS_EVAL_CONFIG")
    for category in ["registered", "casual"]:
        config = load_config(config_file_path)
        config['hf_repo'] = os.path.join(config['hf_repo'], category)
        config['category'] = category
        setup_data, test_data = load_and_split_dataset(backtest_config=config)
        forecast = mk_forecasts(setup_data, config)
        mk_viz(test_data, forecast, config)
