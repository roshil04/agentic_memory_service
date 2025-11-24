from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
import os
import uuid

dates = ["01/08/2022", "01/08/2022", "02/08/2022", "02/08/2022"]
quantities = [57, 43, 51, 66] 
skus = ["PROD-023","PROD-026","PROD-027","PROD-020"]
brands = ["Samsung","Sony","Apple","Sony"]
regions = ["West","North","North","East"]
categories = ["Computers","Electronics","Electronics","Electronics"]

# Convert dates
dates_dt = [datetime.strptime(d, "%d/%m/%Y") for d in dates]

# Prophet expects a DataFrame with ds and y
prophet_data = [{"ds": d, "y": y} for d, y in zip(dates_dt, quantities)]

import pandas as pd
df = pd.DataFrame(prophet_data)

model = Prophet()
model.fit(df)


def forecasting_demand(period: int):
    # Forecast future
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    os.makedirs("predictions", exist_ok=True)

    fig = model.plot(forecast)
    filename = f"predictions/plot_{uuid.uuid4()}.png"

    fig = model.plot(forecast)
    fig.savefig(filename)

    # Build result list (same as your original)
    result = [
        {
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "yhat": row["yhat"],
            "yhat_lower": row["yhat_lower"],
            "yhat_upper": row["yhat_upper"]
        }
        for _, row in forecast.iterrows()
    ]
    
    return result

