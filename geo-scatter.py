import pandas as pd
import plotly.express as px

# Load the CSV file into a DataFrame
file_path = '/Users/lx/Development/timeseries-modeling/data/amazon_delivery.csv'  # replace with your CSV file path
df = pd.read_csv(file_path)
# df['Store_Latitude'] *= -1
# df['Store_Longitude'] *= -1

# Create the scatter geo plot
fig = px.scatter_geo(df,
                    lat="Store_Latitude",
                    lon="Store_Longitude",
                    title="Store Locations")

# Display the plot
fig.show()
# fig.write_html('geo-scatter.html')
