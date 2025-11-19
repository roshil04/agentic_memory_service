import pandas as pd
from prophet import Prophet

# Sample data with 'ds' (datestamp) and 'y' (value) columns
df = pd.read_csv(
    '/home/fm-pc-lt-174/Downloads/AI studio/2025_updated_synth_data_rw.csv',
    parse_dates=["Date"],
    dayfirst=True
)

# Keep only Date and Sales columns
df = df[["Date", "Sales"]]

# Rename for forecasting
df.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)

def forcasting_demand(period:int):

    # df['Date'] = pd.to_datetime(df['Date']) 

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=period)

    forecast = model.predict(future)

        # Convert forecast to list of dicts with JSON-serializable dates
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

    # Convert ds column to ISO strings
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')

    # Convert to list of dicts
    result = result.to_dict(orient='records')

    return result