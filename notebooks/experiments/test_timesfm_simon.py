# %%

import numpy as np
import torch
from transformers import TimesFmModelForPrediction

import altair as alt
import pandas as pd

model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
step, number = 0.1, 250
wave_number = 3
t = 0 + step * np.arange(number)
f = 2 * np.pi * wave_number / t[-1]
t_pred = t[-1] + step * (1 + np.arange(model.config.horizon_length))

def ground_truth(t):
    return np.exp(-0.05 * t) * 10 * np.sin(f * t) + 5 * np.sin(3*f*t)

# Create dummy inputs
forecast_input = [ground_truth(t)]
frequency_input = [0]

# Convert inputs to sequence of tensors
forecast_input_tensor = [
    torch.tensor(ts, dtype=torch.bfloat16).to(model.device) for ts in forecast_input
]
frequency_input_tensor = torch.tensor(frequency_input, dtype=torch.long).to(
    model.device
)

# Get predictions from the pre-trained model
with torch.no_grad():
    outputs = model(
        past_values=forecast_input_tensor, freq=frequency_input_tensor, return_dict=True
    )
    point_forecast_conv = outputs.mean_predictions.float().cpu().numpy()
    quantile_forecast_conv = outputs.full_predictions.float().cpu().numpy()

# plot
base_data = pd.DataFrame({'x': t,
                     'y': forecast_input[0]})
gt_data = pd.DataFrame({'x': t_pred,
                        'y': ground_truth(t_pred)})
data = pd.DataFrame({'x': t_pred,
                     'y': point_forecast_conv[0]})

base = alt.Chart(base_data).mark_line().encode(x='x', y='y')
truth = alt.Chart(gt_data).mark_line(color='red').encode(x='x', y='y')
predict = alt.Chart(data).mark_point().encode(x='x', y='y')

(base + truth + predict).interactive()
# %%


