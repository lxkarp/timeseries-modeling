from evaluate import generate_sample_forecasts, ChronosPipeline, load_and_split_dataset
from metrics import EMD
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.model.evaluation import evaluate_forecasts

import os
import yaml
from pprint import pprint as print


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


def load_config(config_file_path : str):
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
    context = [item for item in context]

    context_by_category = {'casual': context[0][0], 'registered': context[1][0]}
    actuals_by_category = {'casual': context[0][1], 'registered': context[1][1]}
    forecasts_by_category = {'casual': forecast[0], 'registered': forecast[1]}

    for cat, cat_data in context_by_category.items():
        context_data_length = len(cat_data['target'])
        context_data_start = cat_data['start'].to_timestamp()

        plot_dates = pd.date_range(start=context_data_start, periods=config['prediction_length']+context_data_length, freq='D')
        fig, ax = plt.subplots()

        ax.plot(plot_dates, np.append(cat_data['target'], actuals_by_category[cat]['target']))
        forecasts_by_category[cat].plot(ax=ax, show_label=True)
        fig.autofmt_xdate()
        plt.suptitle(f'Chronos {cat} {config["segment_name"]} Forecasts', fontsize=18)
        plt.title('metrics: EMD:{EMD}, MASE:{MASE}, WQL:{WQL}'.format(**metrics), fontsize=10, y=1)
        plt.legend()
        plt.savefig(f'./chronos_{cat}_{config["segment_name"]}.png')


if __name__ == '__main__':
    config_file_path = os.environ.get('CHRONOS_EVAL_CONFIG')
    if config_file_path is None:
        raise ValueError("You must set the environment variable $CHRONOS_EVAL_CONFIG")
    config = load_config(config_file_path)
    test_data = load_and_split_dataset(backtest_config=config)
    forecast = mk_forecasts(test_data)
    mk_viz(test_data, forecast, config)
    print(mk_metrics(test_data, forecast))
