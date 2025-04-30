## Discussion

**Why do all of these models seem to get worse quality predictions with larger context ratios?**
- Longer context lengths can sometimes amplify noise or irrelevant patterns in the data. When the model tries to make predictions based on this noisy context, it may produce less reliable forecasts.
- Longer context lengths can amplify the impact of concept drift. As the model tries to capture a broader range of historical patterns, it might struggle to adapt to the changing dynamics, especially when the prediction length remains constant.
- In some cases, longer context lengths can lead to overfitting, especially if the model is complex. Data Imbalance: With a higher context-to-prediction ratio, the model is exposed to an imbalanced dataset. The model might overfit to the context data, leading to poor generalization and reduced performance on the prediction task.
- Lack of Generalization: The model might become overly specific to the context data, failing to generalize to the prediction task. Put another way, the increased context data might be inconsistent with the actuals in the prediction window. 
- Evaluation Metric Sensitivity: The choice of evaluation metrics can also influence the perceived performance. Some metrics might be more sensitive to changes in the context-to-prediction ratio, leading to a perceived decrease in performance
	- WQL's sensitivity to the context-to-prediction ratio can arise when the distribution of errors changes with varying ratios. For example, if the context length increases, the distribution of errors might shift, leading to different quantile values and, consequently, different WQL values. This can make it challenging to directly compare WQL values across different ratios, especially if the underlying data patterns are complex or non-stationary. [citation needed]


**What do good scores on our error metrics mean? What does it mean for WQL, EMD or MASE to be low, respectively?**

Low mWQL suggests that the model can capture the distribution of errors effectively, regardless of the quantile level.

A good score on mWQL means:
- The model's forecasts are consistently accurate across different quantile levels. 
- The model can handle a variety of error distributions and is robust to changes in the data's underlying patterns.
- The model's performance is well-calibrated, ensuring that the forecasts are not overly optimistic or pessimistic.

A good score on MASE means:
- The model's forecasts are accurate and reliable.
- The model can capture the underlying patterns and trends in the time series data effectively.
- The model's performance is consistent and robust, even when the data exhibits non-stationary behavior or concept drift.

MASE is particularly useful when comparing the performance of different forecasting models, as it provides a standardized measure of error that is independent of the scale of the data.


A low EMD value indicates that the model's forecasts are not only accurate at the mean or median but also capture the spread and shape of the actual values accurately. This metric is valuable for assessing the calibration and overall performance of probabilistic forecasting models.

A low EMD value means:
- The model's forecasts are well-calibrated and closely match the actual values.
- The model can effectively capture the variability and uncertainty in the data, resulting in forecasts that are not overly optimistic or pessimistic.
- The model's performance is robust and consistent across different quantile levels, indicating its ability to handle a range of error scenarios.
- The forecasts are reliable and can be trusted for decision-making purposes, as they closely resemble the actual outcomes.


**What classes of products // styles of behavior do our registered and casual categories show?**


**Other observations**
Prophet underperforms everything
Chronos really suffers with short prediction lengths week10
Naive over performs for short prediction lengths

#### Future Work
Now that we have a test bench, we can easily evaluate other models, including other Chronos sizes. 

Gapped prediction TK

How hard is it to train other models using the Chonos code? What if instead of T5 as a base model I wanted to use one of the updates google has released (byT5 - trained on byte sequences rather than SentencePiece subword tokens, UL2 - trained with various denoising objectives, etc) or something that changes something within the transformer architecture (nystromformer - an optimization of self-attention, KAT - Kolmogrov-Arnold Transformer)?

## Discussion

**Why do all of these models seem to get worse quality predictions with larger context ratios?**
- Longer context lengths can sometimes amplify noise or irrelevant patterns in the data. When the model tries to make predictions based on this noisy context, it may produce less reliable forecasts.
- Longer context lengths can amplify the impact of concept drift. As the model tries to capture a broader range of historical patterns, it might struggle to adapt to the changing dynamics, especially when the prediction length remains constant.
- In some cases, longer context lengths can lead to overfitting, especially if the model is complex. Data Imbalance: With a higher context-to-prediction ratio, the model is exposed to an imbalanced dataset. The model might overfit to the context data, leading to poor generalization and reduced performance on the prediction task.
- Lack of Generalization: The model might become overly specific to the context data, failing to generalize to the prediction task. Put another way, the increased context data might be inconsistent with the actuals in the prediction window. 
- Evaluation Metric Sensitivity: The choice of evaluation metrics can also influence the perceived performance. Some metrics might be more sensitive to changes in the context-to-prediction ratio, leading to a perceived decrease in performance
	- WQL's sensitivity to the context-to-prediction ratio can arise when the distribution of errors changes with varying ratios. For example, if the context length increases, the distribution of errors might shift, leading to different quantile values and, consequently, different WQL values. This can make it challenging to directly compare WQL values across different ratios, especially if the underlying data patterns are complex or non-stationary. [citation needed]


**What do good scores on our error metrics mean? What does it mean for WQL, EMD or MASE to be low, respectively?**

Low mWQL suggests that the model can capture the distribution of errors effectively, regardless of the quantile level.

A good score on mWQL means:
- The model's forecasts are consistently accurate across different quantile levels. 
- The model can handle a variety of error distributions and is robust to changes in the data's underlying patterns.
- The model's performance is well-calibrated, ensuring that the forecasts are not overly optimistic or pessimistic.

A good score on MASE means:
- The model's forecasts are accurate and reliable.
- The model can capture the underlying patterns and trends in the time series data effectively.
- The model's performance is consistent and robust, even when the data exhibits non-stationary behavior or concept drift.

MASE is particularly useful when comparing the performance of different forecasting models, as it provides a standardized measure of error that is independent of the scale of the data.


A low EMD value indicates that the model's forecasts are not only accurate at the mean or median but also capture the spread and shape of the actual values accurately. This metric is valuable for assessing the calibration and overall performance of probabilistic forecasting models.

A low EMD value means:
- The model's forecasts are well-calibrated and closely match the actual values.
- The model can effectively capture the variability and uncertainty in the data, resulting in forecasts that are not overly optimistic or pessimistic.
- The model's performance is robust and consistent across different quantile levels, indicating its ability to handle a range of error scenarios.
- The forecasts are reliable and can be trusted for decision-making purposes, as they closely resemble the actual outcomes.


**What classes of products // styles of behavior do our registered and casual categories show?**


**Other observations**
Prophet underperforms everything
Chronos really suffers with short prediction lengths week10
Naive over performs for short prediction lengths

#### Future Work
Now that we have a test bench, we can easily evaluate other models, including other Chronos sizes. 

Gapped prediction TK

How hard is it to train other models using the Chonos code? What if instead of T5 as a base model I wanted to use one of the updates google has released (byT5 - trained on byte sequences rather than SentencePiece subword tokens, UL2 - trained with various denoising objectives, etc) or something that changes something within the transformer architecture (nystromformer - an optimization of self-attention, KAT - Kolmogrov-Arnold Transformer)?

