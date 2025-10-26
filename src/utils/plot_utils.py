import plotly.graph_objs as go
import pandas as pd

def build_gold_chart(historical_data, prediction_result, start_date, end_date):
    """
    historical_data: dict with 'dates' and 'prices'
    prediction_result: result from predict_future_prices
    start_date, end_date: for slicing historical
    Returns: HTML div of Plotly chart
    """
    df_hist = pd.DataFrame({
        "Date": pd.to_datetime(historical_data["dates"]),
        "Close": historical_data["prices"]
    })

    pred_df = pd.DataFrame({
        "Date": pd.to_datetime(prediction_result["dates"]),
        "Pred_Close": prediction_result["predictions"]
    })

    mask = (df_hist["Date"] >= pd.to_datetime(start_date)) & (df_hist["Date"] <= pd.to_datetime(end_date))
    view = df_hist.loc[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view['Date'], y=view['Close'], mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Pred_Close'], mode='lines+markers', name='Predicted Close'))
    fig.add_trace(go.Scatter(x=view['Date'], y=view['Close'].rolling(30).mean(), mode='lines', name='30d MA'))
    fig.update_layout(title="Gold Price (Close) & Predictions", xaxis_title="Date", yaxis_title="USD")
    from plotly.offline import plot
    return plot(fig, output_type='div')
