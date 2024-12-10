import os
import yaml
import warnings
import numpy as np
from typing import Dict, Tuple, Type

from chronos_utils import load_and_split_dataset
from metrics import mk_viz, save_metrics_to_csv
from gluonts.evaluation import make_evaluation_predictions
from predictors import (
    make_oracle_predictions,
    SeasonalNaivePredictor,
    TrueOraclePredictor,
    OffsetOraclePredictor,
    ConstantMeanPredictor,
    ConstantMedianPredictor,
    ConstantZeroPredictor,
    SkewedMeanPredictor,
    ARIMAPredictor,
)


# Configuration
CONFIG_FILE_PATH = os.environ.get(
    "CONFIG_FILE_PATH", "./evaluation_configs/bike-zero-shot.yaml"
)
DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH", "./data/")
RESULTS_FILE_PATH = os.environ.get("RESULTS_FILE_PATH", "./out/results_metrics.csv")


if CONFIG_FILE_PATH == "./evaluation_configs/bike-zero-shot.yaml":
    warnings.warn("Using default $CONFIG_FILE_PATH")
if DATA_DIR_PATH == "./data/":
    warnings.warn("Using default $DATA_DIR_PATH")
if RESULTS_FILE_PATH == "./out/results_metrics.csv":
    warnings.warn("Using and resetting default $RESULTS_FILE_PATH")
    if os.path.exists(RESULTS_FILE_PATH):
        os.remove(RESULTS_FILE_PATH)

# Model configurations
MODELS: Dict[str, Tuple[Type, int]] = {
    "Naive": (SeasonalNaivePredictor, 1),
    "Oracle": (TrueOraclePredictor, 1),
    "OffsetOracle": (OffsetOraclePredictor, 1),
    "ConstantMean": (ConstantMeanPredictor, 1),
    "ConstantMedian": (ConstantMedianPredictor, 1),
    "ConstantZero": (ConstantZeroPredictor, 1),
    "SkewedAboveMean": (SkewedMeanPredictor, 20),
    "SkewedBelowMean": (SkewedMeanPredictor, 20),
    "ARIMA": (ARIMAPredictor, 20),
}


def load_config(config_file_path: str, segment_name: str) -> dict:
    with open(config_file_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    config = next(
        (cfg for cfg in backtest_configs if cfg["segment_name"] == segment_name), None
    )
    if config is None:
        raise ValueError(f"No configuration found for segment name: {segment_name}")
    return config


def mk_forecasts(data, pipeline, num_samples: int, config: dict):
    if "Oracle" in config["model_name"]:
        forecast_it, _ = make_oracle_predictions(
            dataset=data,
            predictor=pipeline,
            num_samples=num_samples,
        )
        return list(forecast_it)
    else:
        forecast_it, _ = make_evaluation_predictions(
            dataset=data,
            predictor=pipeline,
            num_samples=num_samples,
        )
        return list(forecast_it)


def run_forecasting(
    config: dict, model_name: str, model_predictor: Type, num_samples: int
):
    setup_data, test_data = load_and_split_dataset(backtest_config=config)

    if model_name:
        pipeline = model_predictor(
            prediction_length=np.abs(config["offset"]),
            **(
                {"season_length": config["expected_seasonality"]}
                if model_name == "Naive"
                else {}
            ),
            **({"num_samples": num_samples} if model_name != "Naive" else {}),
            **({"skewness": -10} if "Below" in model_name else {}),
        )

    forecast = mk_forecasts(setup_data, pipeline, num_samples, config)
    metrics = mk_viz(test_data, forecast, config)
    save_metrics_to_csv(metrics, config, RESULTS_FILE_PATH)


def main():
    data_length_multiplier = 4
    segment_config = "july"
    for category in ["registered", "casual"]:
        config = load_config(CONFIG_FILE_PATH, segment_config)
        config["hf_repo"] = os.path.join(
            DATA_DIR_PATH,
            f"ratio_{data_length_multiplier}",
            f"bike_day_{segment_config}",
            category,
        )
        config["prediction_ratio"] = f"{data_length_multiplier - 1}:1"
        config["category"] = category
        print(f"Running {segment_config} - {category} - {config['prediction_ratio']}")

        for model_name, (model_predictor, num_samples) in MODELS.items():
            config["model_name"] = model_name
            print(f"Running {model_name} model")
            run_forecasting(config, model_name, model_predictor, num_samples)


if __name__ == "__main__":
    main()
