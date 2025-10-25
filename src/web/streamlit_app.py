import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from datetime import datetime
from pathlib import Path

DATA_CSV = Path("../GoldLens-AI/data/raw/gold_daily.csv")

st.title("GoldLens AI — Forecasts")
df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
df = df.sort_values("Date")

# Controls
model_choice = st.selectbox("Model", ["ensemble","lstm","bilstm","gru"])
horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)
start_date = st.date_input("Start date", value=df['Date'].max().date() - pd.Timedelta(days=365))
end_date = st.date_input("End date", value=df['Date'].max().date())

# Date range selector
mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
view = df.loc[mask]

# Call local API
if st.button("Predict"):
    payload = {"horizon": int(horizon), "model": model_choice}
    r = requests.post("http://localhost:8000/predict", json=payload)
    result = r.json()
    preds = result['predictions']
    # create date axis for preds
    last_date = view['Date'].max()
    pred_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(preds), freq='D')
    pred_df = pd.DataFrame({"Date": pred_dates, "Pred_Close": preds})

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view['Date'], y=view['Close'], mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Pred_Close'], mode='lines+markers', name='Predicted Close'))
    # add MA line
    fig.add_trace(go.Scatter(x=view['Date'], y=view['Close'].rolling(30).mean(), mode='lines', name='30d MA'))
    fig.update_layout(title="Gold Price (Close) & Predictions", xaxis_title="Date", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)
