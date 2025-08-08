import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------- Configurable Parameters --------
NUM_DAYS = 1                # Change number of days to predict
HOURS_PER_DAY = 24          # Change hours per day (e.g. 24, 48, etc.)
DISPLAY_OPTION = "both"     # Options: "predicted", "both"

# -------- Paths --------
data_folder = r"C:\Users\anand\Documents\energy prediction\data"
comparison_folder = r"C:\Users\anand\Documents\energy prediction\comparison"
actual_file = os.path.join(comparison_folder, "actual_energy.xlsx")
forecast_file = os.path.join(comparison_folder, "next_day_forecast.csv")

# -------- Step 1: Load and Combine Excel Files --------
excel_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.xlsx', '.xls'))]

if not excel_files:
    raise FileNotFoundError(f"No Excel files found in {data_folder}")

all_data = []
for file in excel_files:
    try:
        df = pd.read_excel(file, skiprows=1)
        df.columns = ['Date', 'Hour', 'Energy']
        df['ds'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hour'].astype(str))
        df['y'] = df['Energy']
        df = df[['ds', 'y']]
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

combined_df = pd.concat(all_data).sort_values('ds').reset_index(drop=True)

# -------- Step 2: Train Prophet --------
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
model.fit(combined_df)

# -------- Step 3: Predict Future --------
last_timestamp = combined_df['ds'].max()
total_hours = NUM_DAYS * HOURS_PER_DAY

future = model.make_future_dataframe(periods=total_hours, freq='H')
future = future[future['ds'] > last_timestamp]
forecast = model.predict(future)

# -------- Step 4: Save Forecast CSV (overwrite) --------
os.makedirs(comparison_folder, exist_ok=True)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file, index=False)
print(f"Forecast saved to '{forecast_file}'")

# -------- Step 5: Load Actual Energy Data --------
if not os.path.exists(actual_file):
    print(f"⚠️  Actual energy file not found: {actual_file}")
    actual_df = pd.DataFrame()
else:
    actual_df = pd.read_excel(actual_file, skiprows=1)
    actual_df.columns = ['Date', 'Hour', 'Actual']
    actual_df['ds'] = pd.to_datetime(actual_df['Date'].astype(str) + ' ' + actual_df['Hour'].astype(str))
    actual_df = actual_df[['ds', 'Actual']]

# -------- Step 6: Merge Actual + Predicted --------
if not actual_df.empty:
    merged = pd.merge(forecast[['ds', 'yhat']], actual_df, on='ds', how='inner')
    # -------- Step 7: Calculate Accuracy --------
    merged['Error'] = merged['yhat'] - merged['Actual']
    merged['Absolute_Error'] = merged['Error'].abs()
    merged['APE'] = (merged['Absolute_Error'] / merged['Actual'].replace(0, 1)) * 100  # Avoid div by zero

    mape = merged['APE'].mean()
    accuracy = 100 - mape

    print(f"\nOverall Prediction Accuracy: {accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
else:
    merged = forecast[['ds', 'yhat']].copy()
    print("\n⚠️  Skipping accuracy calculation - actual data not found.")

# -------- Step 8: Plot --------
plt.figure(figsize=(12, 5))

if DISPLAY_OPTION == "predicted":
    plt.plot(merged['ds'], merged['yhat'], marker='o', linestyle='-', label='Predicted')
    plt.title(f"Predicted Energy for {merged['ds'].dt.date.min().strftime('%b %d')}")
elif DISPLAY_OPTION == "both" and 'Actual' in merged.columns:
    plt.plot(merged['ds'], merged['yhat'], marker='o', linestyle='-', label='Predicted')
    plt.plot(merged['ds'], merged['Actual'], marker='s', linestyle='--', label='Actual')
    plt.title(f"Actual vs Predicted Energy for {merged['ds'].dt.date.min().strftime('%b %d')}")
else:
    print(f"\n⚠️ DISPLAY_OPTION set to '{DISPLAY_OPTION}', but no actual data available for comparison.")
    plt.plot(merged['ds'], merged['yhat'], marker='o', linestyle='-', label='Predicted')

plt.xlabel("Hour")
plt.ylabel("Energy Consumed (kVAh)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.tight_layout()
plt.show()
