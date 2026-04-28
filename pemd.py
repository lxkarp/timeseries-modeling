from collections.abc import Iterable, Sequence


def _as_float_list(values: Iterable[float], name: str) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must contain at least one value")
    return converted


def value_distribution_emd(realized: Iterable[float], forecast: Iterable[float]) -> float:
    """
    Wasserstein-1 between empirical distributions of horizon values.

    Both inputs represent one forecast horizon. Time order is ignored: the
    distance is the mean absolute difference between sorted horizon values.
    """
    realized_values = _as_float_list(realized, "realized")
    forecast_values = _as_float_list(forecast, "forecast")

    if len(realized_values) != len(forecast_values):
        raise ValueError("realized and forecast horizons must have the same length")

    sorted_realized = sorted(realized_values)
    sorted_forecast = sorted(forecast_values)
    total_distance = sum(
        abs(actual - predicted)
        for actual, predicted in zip(sorted_realized, sorted_forecast)
    )
    return total_distance / len(sorted_realized)


def probabilistic_value_distribution_emd(
    realized: Iterable[float],
    forecast_samples: Sequence[Iterable[float]],
) -> float:
    """
    Energy-score-style pEMD for a distribution of forecast trajectories.

    The first term is the average value-distribution EMD from each sampled
    trajectory to the realized horizon. The second term is the fair ensemble
    spread correction over ordered non-self pairs.
    """
    realized_values = _as_float_list(realized, "realized")
    samples = [
        _as_float_list(sample, f"forecast sample {sample_index}")
        for sample_index, sample in enumerate(forecast_samples)
    ]

    if not samples:
        raise ValueError("forecast_samples must contain at least one sample")

    first_term = sum(
        value_distribution_emd(realized_values, sample) for sample in samples
    ) / len(samples)

    if len(samples) == 1:
        return first_term

    pairwise_total = 0.0
    pairwise_count = 0
    for left_index, left_sample in enumerate(samples):
        for right_index, right_sample in enumerate(samples):
            if left_index == right_index:
                continue
            pairwise_total += value_distribution_emd(left_sample, right_sample)
            pairwise_count += 1

    return first_term - 0.5 * (pairwise_total / pairwise_count)
