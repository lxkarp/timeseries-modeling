import os
import yaml
import warnings
import numpy as np
from typing import Dict, Tuple, Type

from chronos_utils import load_and_split_dataset
from metrics import mk_viz, save_metrics_to_csv
from gluonts.evaluation import make_evaluation_predictions
from predictors import (
    ARIMAPredictor,
    ProphetPredictor,
    SeasonalNaivePredictor,
    SeasonalWindowAveragePredictor,
    SCUMForecastPredictor,
)

# Configuration
CONFIG_FILE_PATH = os.environ.get(
    "CONFIG_FILE_PATH", "./evaluation_configs/bike-zero-shot.yaml"
)
DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH", "./data/")
USE_CHRONOS = os.environ.get("USE_CHRONOS", "false").lower() == "true"
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
    # "Naive": (SeasonalNaivePredictor, 1),
    # "Prophet": (ProphetPredictor, 20),
    # "ARIMA": (ARIMAPredictor, 20),
    # "HistoricalMean": (SeasonalWindowAveragePredictor, 20),
    "SCUM": (SCUMForecastPredictor, 20),
}

# Chronos-specific configurations
CHRONOS_CONFIG = {
    "model_id": "amazon/chronos-t5-small",
    "device": "cuda:0",
    "torch_dtype": "bfloat16",
    "batch_size": 32,
}

if USE_CHRONOS:
    try:
        from evaluate import generate_sample_forecasts, ChronosPipeline

        MODELS["Chronos"] = (ChronosPipeline, 20)
        print("Chronos model enabled.")
    except ImportError:
        warnings.warn(
            "Failed to import Chronos dependencies. Chronos model will not be available."
        )


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
    if USE_CHRONOS and isinstance(pipeline, ChronosPipeline):
        return generate_sample_forecasts(
            data,
            pipeline=pipeline,
            prediction_length=config["prediction_length"],
            batch_size=CHRONOS_CONFIG["batch_size"],
            num_samples=num_samples,
            temperature=None,
            top_k=None,
            top_p=None,
        )
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

    if USE_CHRONOS and model_name == "Chronos":
        pipeline = ChronosPipeline.from_pretrained(
            CHRONOS_CONFIG["model_id"],
            device_map=CHRONOS_CONFIG["device"],
            torch_dtype=CHRONOS_CONFIG["torch_dtype"],
        )
    elif model_name == "Prophet":
        pipeline = model_predictor(prediction_length=np.abs(config["offset"]))
    elif model_name == "ARIMA":
        pipeline = model_predictor(
            prediction_length=np.abs(config["offset"]),
            season_length=config["expected_seasonality"],
            quantile_levels=np.linspace(0.1, 0.9, 9).tolist(),
        )
    elif model_name == "SCUM":
        pipeline = model_predictor(
            prediction_length=np.abs(config["offset"]),
            season_length=config["expected_seasonality"],
            quantile_levels=np.linspace(0.1, 0.9, 9).tolist(),
        )
    else:
        pipeline = model_predictor(
            prediction_length=np.abs(config["offset"]),
            season_length=config["expected_seasonality"],
            window_size=config["prediction_length"],
        )

    if USE_CHRONOS and model_name == "Chronos":
        forecast = mk_forecasts(test_data.input, pipeline, num_samples, config)
    else:
        forecast = mk_forecasts(setup_data, pipeline, num_samples, config)
    metrics = mk_viz(test_data, forecast, config)
    save_metrics_to_csv(metrics, config, RESULTS_FILE_PATH)


def main():
    for data_length_multiplier in [3, 4, 5, 6]:
        for segment_config in ["week10", "july", "q4"]:
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
                print(
                    f"Running {segment_config} - {category} - {config['prediction_ratio']}"
                )

                for model_name, (model_predictor, num_samples) in MODELS.items():
                    config["model_name"] = model_name
                    run_forecasting(config, model_name, model_predictor, num_samples)


if __name__ == "__main__":
    main()
