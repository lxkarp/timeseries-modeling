import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('./out/results_metrics.csv')

# Filter the DataFrame for 'casual' category and 'q4' segment
filtered_df = df[
    (df['category'] == 'casual') &
    (df['segment_name'] == 'q4')
].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Extract context length from the 'ratio' column
filtered_df['context_length'] = filtered_df['ratio'].apply(lambda x: int(x.split(':')[0]))

# Ensure 'context_length' is sorted
filtered_df = filtered_df.sort_values(['context_length', 'model_name'])

# List of models to plot
models = ['ARIMA', 'Naive', 'Prophet', 'Chronos']

# Create a color map for the models
color_map = {
    'ARIMA': 'blue',
    'Naive': 'green',
    'Prophet': 'red',
    'Chronos': 'purple'
}

# Create a marker map for the models
marker_map = {
    'ARIMA': 'o',
    'Naive': 's',
    'Prophet': '^',
    'Chronos': 'D'
}

# Plotting
plt.figure(figsize=(10, 6))

for model in models:
    # Get data for the model
    model_df = filtered_df[filtered_df['model_name'] == model]
    # Sort by context length
    model_df = model_df.sort_values('context_length')
    # Plot MASE vs. context length
    plt.plot(
        model_df['context_length'],
        model_df['MASE'],
        marker=marker_map.get(model, 'o'),
        linestyle='-',
        color=color_map.get(model, 'black'),
        label=model
    )

# Add labels and title
plt.xlabel('Context Length (Quarters)')
plt.ylabel('Mean Absolute Scaled Error (MASE)')
plt.title('MASE vs. Context Length for Q4 Casual Data Across Models')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Show plot
plt.show()
