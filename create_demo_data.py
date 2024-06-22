import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sales data
def generate_sales_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    sales = np.random.normal(1000, 200, len(date_range))  # Base sales
    
    # Add trend
    trend = np.linspace(0, 500, len(date_range))
    sales += trend
    
    # Add seasonality
    seasonality = 200 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 365))
    sales += seasonality
    
    # Ensure no negative sales
    sales = np.maximum(sales, 0)
    
    df = pd.DataFrame({'Date': date_range, 'Sales': sales.astype(int)})
    return df

# Generate event data
def generate_event_data(start_date, end_date):
    events = [
        ("New Year's Day", [1]),
        ("Valentine's Day", [2]),
        ("Easter", [4]),  # Simplified, actually varies
        ("Mother's Day", [5]),
        ("Father's Day", [6]),
        ("Independence Day", [7]),
        ("Black Friday", [11]),
        ("Christmas", [12]),
        ("Summer Sale", [6, 7, 8]),
        ("Winter Sale", [1, 2])
    ]
    
    event_list = []
    current_date = start_date
    while current_date <= end_date:
        for event_name, months in events:
            if current_date.month in months:
                if event_name in ["Summer Sale", "Winter Sale"]:
                    # Sales last for a week
                    for i in range(7):
                        event_date = current_date + timedelta(days=i)
                        if event_date <= end_date:
                            event_list.append((event_date, event_name))
                else:
                    event_list.append((current_date, event_name))
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(event_list, columns=['Date', 'Event'])
    return df

# Generate data
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 12, 31)

sales_df = generate_sales_data(start_date, end_date)
events_df = generate_event_data(start_date, end_date)

# Save to CSV
sales_df.to_csv('demo_sales_data.csv', index=False)
events_df.to_csv('demo_events_data.csv', index=False)

print("Demo data generated and saved to 'demo_sales_data.csv' and 'demo_events_data.csv'")

# Display sample of the generated data
print("\nSample of sales data:")
print(sales_df.head())

print("\nSample of events data:")
print(events_df.head())

print("\nData range:")
print(f"Start date: {sales_df['Date'].min()}")
print(f"End date: {sales_df['Date'].max()}")
print(f"Total days: {len(sales_df)}")

print("\nUnique events:")
print(events_df['Event'].unique())
