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
        .stack(level=0)
        .reindex(columns=micolumns)
    )


def dataframe_to_latex(df):
    """
    Convert a multi-index DataFrame to a LaTeX tabular format string.

    Parameters:
    df (pd.DataFrame): The multi-index DataFrame to convert.

    Returns:
    str: The LaTeX tabular format string.
    """
    return df.to_latex(multicolumn=True, multirow=True)


# Usage
df = load_csv_as_dataframe("./out/result_metrics.csv")
print(df.head())
df_groups = group_by_ratio(df)
for _, group in df_groups.items():
    print(dataframe_to_latex(create_multiindex(group)))
    print("-------------------")
