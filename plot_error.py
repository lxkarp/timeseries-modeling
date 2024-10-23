import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('./out/results_metrics.csv')

# Filter the DataFrame for ARIMA model, 'casual' category, and 'q4' segment
filtered_df = df[
    (df['model_name'] == 'ARIMA') &
    (df['category'] == 'casual') &
    (df['segment_name'] == 'q4')
]

# Extract context length from the 'ratio' column
filtered_df = filtered_df.copy()
filtered_df['context_length'] = filtered_df['ratio'].apply(lambda x: int(x.split(':')[0]))

# Sort the DataFrame by context length
filtered_df = filtered_df.sort_values('context_length')

# Plot MASE vs. context length
plt.figure(figsize=(8, 6))
plt.plot(filtered_df['context_length'], filtered_df['MASE'], marker='o', linestyle='-')
plt.xlabel('Context Length (Quarters)')
plt.ylabel('Mean Absolute Scaled Error (MASE)')
plt.title('MASE vs. Context Length for ARIMA on Q4 Casual Data')
plt.grid(True)
plt.show()
