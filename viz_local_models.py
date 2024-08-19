from chronos_utils import load_and_split_dataset
from metrics import mk_viz, mk_metrics

import numpy as np

from predictors import ARIMAPredictor, ProphetPredictor, SeasonalNaivePredictor
from gluonts.evaluation import make_evaluation_predictions

import os
import yaml

# from pprint import pprint as print

models = {
    "Naive": (SeasonalNaivePredictor, 1),
    "Prophet": (ProphetPredictor, 20),
    "ARIMA": (ARIMAPredictor, 20),
    # "Chronos": ChronosPredictor,
}


def load_config(config_file_path: str):
    with open(config_file_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    # !!! you can only have a single config in your yaml
    config = backtest_configs[0]
    return config


def mk_forecasts(test_data, pipeline, num_samples):
    forecast_it, _ = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=pipeline,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )

    sample_forecasts = list(forecast_it)

    return sample_forecasts


if __name__ == "__main__":
    config_file_path = os.environ.get("CHRONOS_EVAL_CONFIG")
    if config_file_path is None:
        raise ValueError("You must set the environment variable $CHRONOS_EVAL_CONFIG")

    for category in ["registered", "casual"]:
        config = load_config(config_file_path)
        config["hf_repo"] = os.path.join(config["hf_repo"], category)
        config["category"] = category
        setup_data, test_data = load_and_split_dataset(backtest_config=config)
        for model_name, (model_predictor, num_samples) in models.items():
            config["model_name"] = model_name
            if model_name == "Naive":
                pipeline = model_predictor(
                    prediction_length=np.abs(config["offset"]),
                    season_length=config["expected_seasonality"],
                )
            else:
                pipeline = model_predictor(prediction_length=np.abs(config["offset"]))
            forecast = mk_forecasts(setup_data, pipeline, num_samples)
            mk_viz(test_data, forecast, config)
