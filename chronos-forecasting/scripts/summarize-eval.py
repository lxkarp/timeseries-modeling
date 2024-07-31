import pandas as pd
from scipy.stats import gmean  # requires: pip install scipy


def agg_relative_score(model_df: pd.DataFrame, baseline_df: pd.DataFrame):
    relative_score = model_df.drop("model", axis="columns") / baseline_df.drop(
        "model", axis="columns"
    )
    return relative_score.agg(gmean)


result_df = pd.read_csv("evaluation/results/chronos-t5-small-in-domain.csv").set_index("dataset")
baseline_df = pd.read_csv("evaluation/results/seasonal-naive-in-domain.csv").set_index("dataset")

agg_score_df = agg_relative_score(result_df, baseline_df)
print(agg_score_df)
