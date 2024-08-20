from evaluate import generate_sample_forecasts, ChronosPipeline
from chronos_utils import load_and_split_dataset
from metrics import mk_metrics, mk_viz
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

if __name__ == '__main__':
    config_file_path = os.environ.get('CHRONOS_EVAL_CONFIG')
    if config_file_path is None:
        raise ValueError("You must set the environment variable $CHRONOS_EVAL_CONFIG")

    for category in ["registered", "casual"]:
        config = load_config(config_file_path)
        config["hf_repo"] = os.path.join(config["hf_repo"], category)
        config["category"] = category
        config["model_name"] = "Chronos"
        setup_data, test_data = load_and_split_dataset(backtest_config=config)

        forecast = mk_forecasts(test_data)
        mk_viz(test_data, forecast, config)
