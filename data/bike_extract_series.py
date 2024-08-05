import pandas as pd

# this is actually relative to where you *run* the script so BEWARE
data_file = "./bike_day.csv"

if __name__ == '__main__':
    df = pd.read_csv(data_file, parse_dates=['dteday'])

    # munge all the dtedays into datetinme objects before indexing on them
    df = df.set_index(['dteday'])

    july_casual       = df.loc['2011-04-01':'2011-07-31']['casual']
    july_registered   = df.loc['2011-04-01':'2011-07-31']['registered']
    july_casual.to_csv("./bike_july_casual.csv")
    july_registered.to_csv("./bike_july_registered.csv")

    week10_casual     = df.loc['2011-02-13':'2011-06-12']['casual']
    week10_registered = df.loc['2011-02-13':'2011-06-12']['registered']
    week10_casual.to_csv("./bike_week10_casual.csv")
    week10_registered.to_csv("./bike_week10_registered.csv")

    q4_casual         = df.loc['2011-09-01':'2012-11-30']['casual']
    q4_registered     = df.loc['2011-09-01':'2012-11-30']['registered']
    q4_casual.to_csv("./bike_q4_casual.csv")
    q4_registered.to_csv("./bike_q4_registered.csv")

