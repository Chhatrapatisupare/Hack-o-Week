import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_mock_data(filename='proximity_data.csv'):
    """Generate mock Kaggle time-series proximity data."""
    print("Generating mock time-series proximity data...")
    timestamps = [datetime.now() - timedelta(days=x) for x in range(100, 0, -1)]
    # Simulate some distance trends
    distances = np.linspace(10, 50, 100) + np.random.normal(0, 5, 100)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'distance': distances
    })
    df.to_csv(filename, index=False)
    print(f"Mock data saved to {filename}")
    return filename

def process_and_visualize(filename):
    """Process proximity data, compute averages and plot trends."""
    print(f"Reading data from {filename}...")
    df = pd.read_csv(filename)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Compute averages
    average_distance = df['distance'].mean()
    print(f"Overall Average Distance: {average_distance:.2f}")
    
    # Weekly average
    df.set_index('timestamp', inplace=True)
    weekly_avg = df['distance'].resample('W').mean()
    print("\nWeekly Average Distances:")
    print(weekly_avg)
    
    # Plot distance trends
    print("\nPlotting distance trends...")
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['distance'], marker='o', linestyle='-', alpha=0.6, label='Daily Distance')
    plt.plot(weekly_avg.index, weekly_avg, color='red', linewidth=3, label='Weekly Average')
    
    # Add an average line
    plt.axhline(y=average_distance, color='green', linestyle='--', label=f'Overall Avg ({average_distance:.2f})')
    
    plt.title('Time-Series Proximity Data: Distance Trends', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('distance_trends.png')
    print("Plot saved as 'distance_trends.png'.")

if __name__ == "__main__":
    # Generate mock data since an actual Kaggle dataset wasn't provided
    data_file = generate_mock_data()
    
    # Process the data, compute averages, and visualize
    process_and_visualize(data_file)
