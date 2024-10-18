import pandas as pd
import numpy as np


def load_csv_as_dataframe(file_path):
    """
    Load a CSV file as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)


def group_by_ratio(df):
    """
    Group the DataFrame by the 'ratio' column and return the groups as a dictionary of DataFrames.

    Parameters:
    df (pd.DataFrame): The DataFrame to group.

    Returns:
    dict: A dictionary where the keys are the unique values from the 'ratio' column and the values are DataFrames corresponding to each group, with the 'ratio' column dropped.
    """
    return {ratio: group.drop(columns="ratio") for ratio, group in df.groupby("ratio")}


def group_by_model(df):
    """
    Group the DataFrame by the 'model_name' column and return the groups as a dictionary of DataFrames.

    Parameters:
    df (pd.DataFrame): The DataFrame to group.

    Returns:
    dict: A dictionary where the keys are the unique values from the 'model_name' column and the values are DataFrames corresponding to each group, with the 'model_name' column dropped.
    """
    return {
        model: group.drop(columns="model_name")
        for model, group in df.groupby("model_name")
    }


def create_multiindex(df):
    """
    This function transforms the input DataFrame into a multi-index DataFrame where the columns
    are a MultiIndex created from the unique values of 'segment_name' and 'category'. The index
    of the resulting DataFrame is 'model_name'.

    df (pd.DataFrame): The DataFrame to transform. It must contain the columns 'model_name',
                       'segment_name', and 'category'.

    pd.DataFrame: The transformed DataFrame with a multi-index on the columns.
    """

    micolumns = pd.MultiIndex.from_product(
        [df["segment_name"].unique(), np.sort(df["category"].unique())]
    )

    return (
        df.pivot(columns=["segment_name", "category"], index=["model_name"])
        .stack(level=0, future_stack=True)
        .reindex(columns=micolumns)
    )


def dataframe_to_latex(df, ratio):
    """
    Convert a multi-index DataFrame to a LaTeX tabular format string.

    Parameters:
    df (pd.DataFrame): The multi-index DataFrame to convert.

    Returns:
    str: The LaTeX tabular format string.
    """
    ret_latex = df.to_latex(
        multicolumn=True,
        multirow=True,
        float_format="%.4f",
        caption=f"Metrics for ratio {ratio}",
    )
    for col in df.columns:
        min_emd_val = df.xs("EMD", level=1)[col].min()
        min_wql_val = df.xs("WQL", level=1)[col].min()
        min_mase_val = df.xs("MASE", level=1)[col].min()

        lines = ret_latex.split("\n")
        for i, line in enumerate(lines):
            if "EMD" in line and f"{min_emd_val:.4f}" in line:
                # bold the lowest EMD value in each column
                lines[i] = line.replace(
                    f"{min_emd_val:.4f}", f"\\textbf{{{min_emd_val:.4f}}}"
                )
            if "WQL" in line and f"{min_wql_val:.4f}" in line:
                # italisize the lowest WQL value in each column
                lines[i] = line.replace(
                    f"{min_wql_val:.4f}", f"\\textit{{\\textbf{{{min_wql_val:.4f}}}}}"
                )
            if "MASE" in line and f"{min_mase_val:.4f}" in line:
                # underline the lowest MASE value in each column
                lines[i] = line.replace(
                    f"{min_mase_val:.4f}",
                    f"\\underline{{\\textbf{{{min_mase_val:.4f}}}}}",
                )
        ret_latex = "\n" + "\n".join(lines)

    ret_latex = (
        ret_latex.replace("{table}", "{table*}")
        .replace("multirow[t]{", "multirow{")
        .replace("Naive", "Seasonal Naïve")
    )

    return ret_latex


# Usage
df = load_csv_as_dataframe("./out/results_metrics.csv")
print(df.head())
df_groups = group_by_ratio(df)
for rat, group in df_groups.items():
    print(dataframe_to_latex(create_multiindex(group), rat))
