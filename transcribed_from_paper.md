# Dataset

Our evaluation focuses on a single dataset comprising bicycle rental records from the Capital Bikeshare program in Washington, D.C., spanning 2011 to 2012. Each data point in the original dataset represents an individual bike rental, including a timestamp accurate to the minute along with various additional attributes.
To prepare the dataset for analysis using Chronos, we reduced the dimensionality to include only the timestamp, and a binary indicator of customer type (casual or registered). Registered customers pay a subscription fee for unlimited rentals, while casual customers encompass all other users.
To ensure uniform intervals between data points, as required by our models, we aggregated the data to a daily basis, summing the total number of bikes rented per day. The source code for this data transformation and aggregation process is available at [URL].

To evaluate model performance across different time scales, we created three subsets of the source data, maintaining a consistent context-to-prediction length ratio of approximately 3:1:

Short-term: 3 weeks context, 1 week prediction
Medium-term: 3 months context, 1 month prediction
Long-term: 9 months context, 3 months prediction

Each of these subsets was further divided into two equal-length subsets based on customer type (registered and casual), resulting in six distinct evaluation subsets.

[ TABLE ]

# Process

Our evaluation process consists of two main stages:

1. Forecasting
2. Visualizing & Summarizing

While our forecasting methodology aligns with Amazon's "zero-shot" evaluation approach, our paper diverges in the visualization and summarization techniques employed.

## Forecasting

Our forecasting process closely follows the methods described in the "zero-shot" evaluation section of the Amazon paper. For Chronos, we selected the amazon/chronos-t5-small model and configured our test harness to use the same hyperparameters as Amazon, specifically for batch_size and num_samples.
The context data was provided to all models:

Prophet and AutoARIMA use this context to tune their parameters.
The underlying T5 model used by Chronos uses this context to set weights and activations.

We evaluated each of our four models (Chronos, Prophet, AutoARIMA, and Naive) against all six of our evaluation datasets, resulting in a total of 24 forecasted outputs.

## Visualizing & Summarizing

To comprehensively assess and compare the performance of our models, we employed several evaluation metrics and visualization techniques:

TODO What and WHY EMD, WQL, MASE

# Questions

- Are we sure that the Capital bike share data isn't anywhere in T5's training data?

- Why did we aggregate to the daily level instead of hourly?

- How do we prove we haven't fucked up setting up Chronos? Do we have any recorded evidence that we get the same values as Amazon does when we run their evaluations against their provided dataset?

- Why did we choose to separate our data by registered/casual? How do we explain or articulate this?

- Are we sure that the way I described registered vs casual above is accurate? i.e what exactly does "registered" mean?

