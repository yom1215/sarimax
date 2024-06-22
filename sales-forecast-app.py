import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from io import StringIO

def load_data(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8')
    df = pd.read_csv(StringIO(content))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def load_events(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8')
    events_df = pd.read_csv(StringIO(content))
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    return events_df

def prepare_exog_variables(sales_df, events_df):
    # Create a DataFrame with the same index as sales_df
    exog_df = pd.DataFrame(index=sales_df.index)
    
    # Create dummy variables for each unique event
    for event in events_df['Event'].unique():
        event_dates = events_df[events_df['Event'] == event]['Date']
        exog_df[event] = exog_df.index.isin(event_dates).astype(int)
    
    return exog_df

def create_sarima_model(data, exog):
    model = SARIMAX(data, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results

def forecast_sales(model, steps, exog_future):
    forecast = model.forecast(steps=steps, exog=exog_future)
    return forecast

def main():
    st.title('Sales Forecast App with Event Data')

    sales_file = st.file_uploader("Choose a CSV file for sales data", type="csv")
    events_file = st.file_uploader("Choose a CSV file for event data", type="csv")

    if sales_file is not None and events_file is not None:
        sales_df = load_data(sales_file)
        events_df = load_events(events_file)

        st.write("Sales Data Preview:")
        st.write(sales_df.head())

        st.write("Events Data Preview:")
        st.write(events_df.head())

        exog_df = prepare_exog_variables(sales_df, events_df)

        model = create_sarima_model(sales_df['Sales'], exog_df)

        forecast_days = 30
        
        # Prepare future exogenous variables
        future_dates = pd.date_range(start=sales_df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        future_exog = pd.DataFrame(index=future_dates)
        for event in exog_df.columns:
            future_event_dates = events_df[events_df['Event'] == event]['Date']
            future_exog[event] = future_exog.index.isin(future_event_dates).astype(int)

        forecast = forecast_sales(model, forecast_days, future_exog)

        # Combine original data and forecast
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
        forecast_df.set_index('Date', inplace=True)
        combined_df = pd.concat([sales_df, forecast_df])

        st.write("Sales Data and Forecast:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sales_df.index, sales_df['Sales'], label='Original Data')
        ax.plot(forecast_df.index, forecast_df['Forecast'], color='red', label='Forecast')
        
        # Plot event markers
        for event in events_df['Event'].unique():
            event_dates = events_df[events_df['Event'] == event]['Date']
            ax.scatter(event_dates, sales_df.loc[event_dates, 'Sales'], marker='o', label=event)

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Sales Data and Forecast with Events')
        ax.legend()
        st.pyplot(fig)

        st.write("Combined Data and Forecast Table:")
        st.write(combined_df)

        # Download button for the combined data
        csv = combined_df.to_csv()
        st.download_button(
            label="Download combined data as CSV",
            data=csv,
            file_name="sales_forecast_data.csv",
            mime="text/csv",
        )

        if st.button("Show Model Details"):
            st.write("Model Summary:")
            summary = model.summary()
            st.write(summary)

            # Download button for model summary
            summary_text = str(summary)
            st.download_button(
                label="Download Model Summary",
                data=summary_text,
                file_name="model_summary.txt",
                mime="text/plain",
            )

if __name__ == "__main__":
    main()
