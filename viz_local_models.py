from chronos_utils import load_and_split_dataset
from metrics import mk_viz

# from chronos import ChronosPipeline

import numpy as np

from predictors import ARIMAPredictor, ProphetPredictor, SeasonalNaivePredictor
from gluonts.evaluation import make_evaluation_predictions

import os
import yaml
import warnings

# from pprint import pprint as print

models = {
    "Naive": (SeasonalNaivePredictor, 1),
    "Prophet": (ProphetPredictor, 20),
    # "ARIMA": (ARIMAPredictor, 20),
    #    "Chronos": (ChronosPipeline, 20),
}


def load_config(config_file_path: str, segment_name: str):
    with open(config_file_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    # # !!! you can only have a single config in your yaml
    # config = backtest_configs[0]
    # Find the config where the 'segment_name' matches the field 'segment_name'
    config = next(
        (cfg for cfg in backtest_configs if cfg["segment_name"] == segment_name), None
    )
    if config is None:
        raise ValueError(f"No configuration found for segment name: {segment_name}")
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
    config_file_path = os.environ.get("CONFIG_FILE_PATH")
    if config_file_path is None:
        config_file_path = "./evaluation_configs/bike-zero-shot.yaml"
        warnings.warn("Using default $CONFIG_FILE_PATH")

    data_dir_path = os.environ.get("DATA_DIR_PATH")
    if data_dir_path is None:
        data_dir_path = "./data/"
        warnings.warn("Using default $DATA_DIR_PATH")

    results_file_path = os.environ.get("RESULTS_FILE_PATH")
    if results_file_path is None:
        results_file_path = "./out/result_metrics.csv"
        warnings.warn("Using and resetting default $RESULTS_FILE_PATH")
        if os.path.exists(results_file_path):
            os.remove(results_file_path)

    for prediction_ratio in [3, 4, 5, 6]:
        for segment_config in ["week10", "july", "q4"]:
            for category in ["registered", "casual"]:
                config = load_config(config_file_path, segment_config)

                # constructs paths like:
                # "./data/ratio_2/bike_day_q4/casual"
                config["hf_repo"] = os.path.join(
                    data_dir_path,
                    f"ratio_{prediction_ratio}",
                    f"bike_day_{segment_config}",
                    category,
                )
                config["prediction_ratio"] = f"{prediction_ratio - 1}:1"

                config["category"] = category
                setup_data, test_data = load_and_split_dataset(backtest_config=config)
                for model_name, (model_predictor, num_samples) in models.items():
                    config["model_name"] = model_name
                    if model_name == "Chronos":
                        pipeline = ChronosPipeline.from_pretrained(
                            "amazon/chronos-t5-small",
                            device_map="cuda:0",
                            torch_dtype="bfloat16",
                        )
                    elif model_name == "Prophet":
                        pipeline = model_predictor(
                            prediction_length=np.abs(config["offset"])
                        )
                    else:
                        pipeline = model_predictor(
                            prediction_length=np.abs(config["offset"]),
                            season_length=config["expected_seasonality"],
                        )
                    forecast = mk_forecasts(setup_data, pipeline, num_samples)
                    mk_viz(test_data, forecast, config)
