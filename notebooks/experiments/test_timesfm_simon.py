# %%

import numpy as np
import torch
from transformers import TimesFmModelForPrediction

import altair as alt
import pandas as pd

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=device,
)
model.config.horizon_length = 128


step, number = 0.1, 250
wave_number = 3
t = 0 + step * np.arange(number)
f = 2 * np.pi * wave_number / t[-1]
t_pred = t[-1] + step * (1 + np.arange(model.config.horizon_length))

def ground_truth(t):
    return np.exp(-0.005 * t) * 10 * np.sin(f * t) + 5 * np.sin(3.2*f*t)

# Create dummy inputs
forecast_input = [ground_truth(t)]
frequency_input = [0, 1, 2, 3, 4]

# Convert inputs to sequence of tensors
forecast_input_tensor = [
    torch.tensor(ts, dtype=torch.bfloat16).to(model.device) for ts in forecast_input
]
frequency_input_tensor = torch.tensor(frequency_input, dtype=torch.long).to(
    model.device
)
print(f"Input tensor shape: {forecast_input_tensor[0].shape}")
print(f"Frequency tensor shape: {frequency_input_tensor.shape}")

# Get predictions from the pre-trained model
with torch.no_grad():
    outputs = model(
        past_values=forecast_input_tensor, freq=frequency_input_tensor, return_dict=True
    )
    point_forecast_conv = outputs.mean_predictions.float().cpu().numpy()
    quantile_forecast_conv = outputs.full_predictions.float().cpu().numpy()

print(f"Output tensor shape: {point_forecast_conv.shape}")
print(f"Output (quantiles) shape: {quantile_forecast_conv.shape}")

# plot
base_data = pd.DataFrame({'x': t,
                        'y': forecast_input[0],
                        'series': 'Input Series'})
gt_data = pd.DataFrame({'x': t_pred,
                        'y': ground_truth(t_pred),
                        'series': 'Ground Truth'})
point_forecast_data = pd.DataFrame({'x': t_pred,
                        'y': point_forecast_conv[0],
                        'series': 'Point Forecast'})
quantile_data_list = []
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i, q in enumerate(quantiles):
    temp_df = pd.DataFrame({
        'x': t_pred,
        'y': quantile_forecast_conv[0, :, i],
        'quantile': f'Q{q}'
    })
    quantile_data_list.append(temp_df)
quantile_data = pd.concat(quantile_data_list)
confidence_data = pd.DataFrame({
    'x': np.concatenate([t_pred, t_pred[::-1]]),
    'y': np.concatenate([
        quantile_forecast_conv[0, :, 0],  # Q0.1
        quantile_forecast_conv[0, ::-1, 8]  # Q0.9 (inversed to close the polygon)
    ])
})


base_chart = alt.Chart(base_data).mark_line().encode(
    x='x:Q', 
    y='y:Q',
    color=alt.Color('series:N', title="Series")
)
truth_chart = alt.Chart(gt_data).mark_line().encode(
    x='x:Q', 
    y='y:Q',
    color=alt.Color('series:N', title="Series")
)
point_forecast_chart = alt.Chart(point_forecast_data).mark_point(size=60).encode(
    x='x:Q', 
    y='y:Q',
    color=alt.Color('series:N', title="Series")
)
quantile_chart = alt.Chart(quantile_data).mark_line(opacity=0.3).encode(
    x='x:Q',
    y='y:Q',
    color='quantile:N'
)
confidence_chart = alt.Chart(confidence_data).mark_area(opacity=0.3, color='lightgray').encode(
    x='x:Q',
    y='y:Q'
)

final_chart = alt.layer(
    confidence_chart,
    base_chart, 
    truth_chart, 
    point_forecast_chart,
    quantile_chart
).properties(
    title="Time Series Forecasting with TimesFM",
    width=800,
    height=400
).interactive()

final_chart




# %%
import matplotlib.pyplot as plt

# Création de la figure
plt.figure(figsize=(12, 6))

# 1. Zone de confiance (entre Q0.1 et Q0.9)
plt.fill_between(t_pred, 
                quantile_forecast_conv[0, :, 0],  # Q0.1
                quantile_forecast_conv[0, :, 8],  # Q0.9
                alpha=0.3, color='gray', label='80% Confidence Interval')

# 2. Série d'entrée
plt.plot(t, forecast_input[0], 'b-', linewidth=2, label='Input Series')

# 3. Vérité terrain
plt.plot(t_pred, ground_truth(t_pred), 'r-', linewidth=2, label='Ground Truth')

# 4. Prévision ponctuelle
plt.plot(t_pred, point_forecast_conv[0], 'go', markersize=4, label='Point Forecast')

# 5. Quantiles (optionnel - pour voir tous les quantiles)
for i, q in enumerate(quantiles):
    if i == 0 or i == len(quantiles)-1:  # Afficher seulement Q0.1 et Q0.9 pour éviter la surcharge
        continue
    plt.plot(t_pred, quantile_forecast_conv[0, :, i], '--', 
             color='purple', alpha=0.2, linewidth=0.5)

# Configuration du plot
plt.title('Time Series Forecasting with TimesFM', fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# Ajustement des limites pour mieux voir la transition
#plt.xlim([t[-50], t_pred[-1]])

plt.tight_layout()
plt.show()

# %%
