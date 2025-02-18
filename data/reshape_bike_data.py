#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from pprint import pprint
import pyarrow as pa
import pyarrow.parquet as pq
import os
import datetime

df = pd.read_csv("./bike_day.csv")
# make dteday into a proper datetime
df['dteday'] = pd.to_datetime(df['dteday'])
# drop everything except timestamp, registered, cnt
df = df.drop(columns=["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "instant", "cnt"])
df = pd.melt(df, id_vars=['dteday'], value_vars=['casual', 'registered']).groupby('variable').agg(list).reset_index()
df.rename(columns={"variable": "id", "dteday": "timestamp", "value": "target"}, inplace=True)
print(df)


# In[2]:


min_date = min(min(df['timestamp']))
max_date = max(max(df['timestamp']))

print(f"Min date: {min_date}")
print(f"Max date: {max_date}")


# In[23]:


def extract_target(df: pd.DataFrame, start: str, end: str):
    extract_df_casual = sorted((ts,targ) for ts,targ in zip(df['timestamp'][0], df['target'][0]) if ts >= pd.to_datetime(start) and ts <= pd.to_datetime(end))
    extract_df_casual = list(map(list, zip(*extract_df_casual)))

    extract_df_registered = sorted((ts,targ) for ts,targ in zip(df['timestamp'][1], df['target'][1]) if ts >= pd.to_datetime(start) and ts <= pd.to_datetime(end))
    extract_df_registered = list(map(list, zip(*extract_df_registered)))

    output_df = df.copy()
    output_df.loc[0, 'timestamp'] = extract_df_casual[0]
    output_df.loc[0, 'target'] = extract_df_casual[1]
    output_df.loc[1, 'timestamp'] = extract_df_registered[0]
    output_df.loc[1, 'target'] = extract_df_registered[1]
    

    return output_df

def adjust_prediction_config(prediction_config, ratio, min_date):
    """
    Adjusts the prediction configuration by modifying the start dates based on a given ratio and minimum date.

    Parameters:
    prediction_config (dict): A dictionary where keys are identifiers and values are dictionaries with 'start' and 'end' date strings.
    ratio (float): A ratio to adjust the interval between start and end dates.
    min_date (str or pd.Timestamp): The minimum allowable start date. If the new start date is earlier than this, it will be set to this date.

    Returns:
    dict: A dictionary with the same keys as prediction_config, but with adjusted 'start' dates.
    """
    adjusted_config = {}
    for key, dates in prediction_config.items():
        start_date = pd.to_datetime(dates['start'])
        end_date = pd.to_datetime(dates['end'])
        interval = (end_date - start_date + datetime.timedelta(days=1)) * ratio
        new_start_date = (end_date + datetime.timedelta(days=1)) - interval
        if new_start_date < min_date:
            new_start_date = min_date
        adjusted_config[key] = {'start': new_start_date.strftime('%Y-%m-%d'), 'end': dates['end']}
    return adjusted_config


# In[24]:


# config = {
#     'july': {'start': '2011-04-01',
#            'end': '2011-07-31'},
#     'week10': {'start':'2011-02-14',
#              'end':'2011-03-13'},
#     'q4': {'start':'2011-09-01',
#          'end':'2012-11-30'}
# }

prediction_config = {
    'week10': {'start':'2011-03-07',
         'end':'2011-03-13'},
    'july': {'start': '2011-07-01',
           'end': '2011-07-31'},
    'q4': {'start':'2012-09-01',
         'end':'2012-11-30'}
}

for ratio in [2, 3, 4, 5, 6]:
    full_data_config = adjust_prediction_config(prediction_config, ratio, min_date)

    for key, dates in full_data_config.items():
        os.makedirs(f'ratio_{ratio}/bike_day_{key}/casual', exist_ok=True)
        os.makedirs(f'ratio_{ratio}/bike_day_{key}/registered', exist_ok=True)
        temp_df = extract_target(df, **dates)
        temp_df.to_parquet(f'ratio_{ratio}/bike_day_{key}/bike_day_{key}.parquet', index=False)
        
        temp_casual = pd.DataFrame(temp_df.loc[0]).T
        temp_registered = pd.DataFrame(temp_df.loc[1]).T

        pq.write_table(pa.Table.from_pandas(temp_casual), f'ratio_{ratio}/bike_day_{key}/casual/bike_day_{key}.parquet')
        pq.write_table(pa.Table.from_pandas(temp_registered), f'ratio_{ratio}/bike_day_{key}/registered/bike_day_{key}.parquet')
        
        print(len(temp_casual['timestamp'][0]))




pprint(prediction_config)






