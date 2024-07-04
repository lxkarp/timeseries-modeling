import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file into a DataFrame
file_path = '/Users/lx/Development/timeseries-modeling/data/amazon_delivery.csv'  # replace with your CSV file path
df = pd.read_csv(file_path)

# Print basic information about the DataFrame
print("DataFrame Info:")
print(df.info())

# Print the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Calculate variance and median for each column
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
variance = df[numeric_columns].var()
median = df[numeric_columns].median()

# Print variance and median
print("\nVariance of each column:")
print(variance)

print("\nMedian of each column:")
print(median)

# Create a bar plot for variance
fig_variance = px.bar(
    x=variance.index,
    y=variance.values,
    title='Variance of Each Dimension',
    labels={'x': 'Dimension', 'y': 'Variance'}
)
fig_variance.show()

# Create a bar plot for median
fig_median = px.bar(
    x=median.index,
    y=median.values,
    title='Median of Each Dimension',
    labels={'x': 'Dimension', 'y': 'Median'}
)
fig_median.show()

# Summary statistics
summary = df.describe().transpose()
print("\nSummary Statistics:")
print(summary)

# Create a summary plot (box plot) for each dimension
fig_box = go.Figure()

for col in numeric_columns:
    fig_box.add_trace(go.Box(y=df[col], name=col))

fig_box.update_layout(
    title='Box Plot of Each Dimension',
    yaxis_title='Values',
    xaxis_title='Dimension'
)

fig_box.show()
fig_box.write_html('data-summary-boxplot.html')
