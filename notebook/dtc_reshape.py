import pandas as pd
df = pd.read_csv("../first.csv")

# extract style from style & color code
df['STYLE'] = df['UNIV_STYLE_COLOR_CD'].str.split('-', expand=True)[0]
df['TRANS_DT'] = pd.to_datetime(df['TRANS_DT'])

# what shape do we want the data in?
# one row per timeseries
# columns: id, timestamp, target 
# timestamp and target should be same length
# shapes: int, list of datetimes, list of floats
df.drop(columns=['ORD_KEY', 'BOW_DATE', 'GTIN', 'UNIV_STYLE_COLOR_CD',
       'UNIV_SZ_CD', 'CHANNEL', 'COUNTRY', 'GEO', 'ZIP_CD', 'PLANT_CD',
       'RETURN_UNITS', 'NET_SLS_UNITS', 'CLEARANCE_IND',
       'GROSS_AMT_USD', 'GROSS_AMT_LC', 'NET_SLS_AMT_USD', 'NET_SLS_AMT_LC',
       'RETURN_AMT_LC', 'RETURN_AMT_USD', 'MSRP_LC', 'MSRP_USD'], inplace=True)
df = df.groupby(['STYLE', 'TRANS_DT'])['GROSS_UNITS'].sum().reset_index()

# Create a complete date range
dt_min = df['TRANS_DT'].min()
dt_max = df['TRANS_DT'].max()
all_dates = pd.date_range(start=dt_min, end=dt_max, freq='D')
all_styles = df['STYLE'].unique()

# Create a dataframe with every combination of style and date
complete_index = pd.MultiIndex.from_product([all_styles, all_dates], names=['STYLE', 'TRANS_DT'])
df_complete = pd.DataFrame(index=complete_index).reset_index()

# Merge with the original dataframe to fill in the missing combinations with 0s
df = pd.merge(df_complete, df, on=['STYLE', 'TRANS_DT'], how='left').fillna(0)

# Group by style and create lists of timestamps and gross units
grouped_df = df.groupby('STYLE').agg({'TRANS_DT': lambda x: list(x), 'GROSS_UNITS': lambda x: list(x)}).reset_index()

# Rename columns for clarity
grouped_df.rename(columns={'STYLE': 'id', 'TRANS_DT': 'timestamp', 'GROSS_UNITS': 'target'}, inplace=True)

# Save the dataframe to a csv file and parquet file
grouped_df.to_csv('dtc_timeseries.csv', index=False)
grouped_df.to_parquet('dtc_timeseries.parquet', index=False)