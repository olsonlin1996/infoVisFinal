
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# --- CONFIGURATION ---
TARGET_DRIVERS = [1, 11, 22]  # Max, Perez, Tsunoda
DRIVER_NAMES = {1: 'Verstappen', 11: 'Perez', 22: 'Tsunoda'}
FEATURES = ['throttle', 'brake', 'n_gear']

def load_data(driver):
    print(f"Loading data for Driver {driver}...")
    try:
        df_car = pd.read_csv(f'data/car_data_{driver}.csv')
        df_laps = pd.read_csv(f'data/laps_{driver}.csv')
    except FileNotFoundError:
        print(f"  Missing files for {driver}")
        return None, None

    # Time format conversion
    df_car['date'] = pd.to_datetime(df_car['date'], format='mixed')
    df_laps['date_start'] = pd.to_datetime(df_laps['date_start'], format='mixed')
    
    df_car.sort_values('date', inplace=True)
    df_car['driver'] = driver
    
    return df_car, df_laps

def analyze_full_race_styles():
    # Extract all laps for each driver and compute average
    driver_averages = []
    FIXED_POINTS = 100  # More points for full race
    COMMON_STEPS = 200
    
    for driver in TARGET_DRIVERS:
        full_data, laps_data = load_data(driver)
        if full_data is None:
            continue
        
        lap_segments = []
        
        # Extract all laps (full race)
        for _, lap in laps_data.iterrows():
            lap_num = int(lap['lap_number'])
            l_start = lap['date_start']
            
            if pd.isna(lap['lap_duration']):
                continue
            
            l_end = l_start + pd.Timedelta(seconds=lap['lap_duration'])
            lap_df = full_data[(full_data['date'] >= l_start) & (full_data['date'] <= l_end)]
            
            if len(lap_df) > 10:  # Need minimum points
                # Resample to fixed points
                t_orig = np.linspace(0, 1, len(lap_df))
                t_new = np.linspace(0, 1, FIXED_POINTS)
                
                resampled_data = {'lap': [lap_num] * FIXED_POINTS}
                
                for feat in FEATURES:
                    resampled_data[feat] = np.interp(t_new, t_orig, lap_df[feat])
                
                resampled_df = pd.DataFrame(resampled_data)
                lap_segments.append(resampled_df)
        
        print(f"Driver {driver} ({DRIVER_NAMES[driver]}): {len(lap_segments)} valid laps")
        
        if not lap_segments:
            continue
        
        # Compute average across all laps
        combined = pd.concat(lap_segments)
        
        # Average each feature across all laps
        avg_features = {}
        for feat in FEATURES:
            # Reshape to (num_laps, FIXED_POINTS)
            feature_matrix = combined.groupby('lap')[feat].apply(list).values
            feature_matrix = np.array([np.array(f) for f in feature_matrix])
            avg_features[feat] = feature_matrix.mean(axis=0)
        
        # Create average dataframe
        avg_df = pd.DataFrame(avg_features)
        avg_df['driver'] = driver
        avg_df['driver_name'] = DRIVER_NAMES[driver]
        
        # Resample to common steps for MDS
        t = np.linspace(0, 1, len(avg_df))
        t_new = np.linspace(0, 1, COMMON_STEPS)
        
        resampled_avg = {'driver': [driver] * COMMON_STEPS}
        for feat in FEATURES:
            resampled_avg[feat] = np.interp(t_new, t, avg_df[feat])
        
        driver_averages.append(pd.DataFrame(resampled_avg))
    
    if not driver_averages:
        print("No valid data.")
        return
    
    # Combine and project to MDS
    combined_avg = pd.concat(driver_averages, ignore_index=True)
    
    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(combined_avg[FEATURES])
    
    # MDS
    print("Running MDS on full race average trajectories...")
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, n_jobs=-1)
    pos = mds.fit_transform(X_scaled)
    
    # Rotate with PCA
    pca = PCA(n_components=2)
    pos = pca.fit_transform(pos)
    if pos[0, 0] > pos[-1, 0]:
        pos[:, 0] *= -1
    
    combined_avg['mds_x'] = pos[:, 0]
    combined_avg['mds_y'] = pos[:, 1]
    
    # Add driver names
    combined_avg['driver_name'] = combined_avg['driver'].map(DRIVER_NAMES)
    
    # Save
    combined_avg.to_csv('output/driver_average_styles.csv', index=False)
    print("Saved data to output/driver_average_styles.csv")
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = {1: '#9D4EDD', 11: '#0096C7', 22: '#FFD60A'}  # Purple, Blue, Yellow
    
    for driver in TARGET_DRIVERS:
        subset = combined_avg[combined_avg['driver'] == driver]
        plt.plot(subset['mds_x'], subset['mds_y'], 
                 c=colors[driver], label=DRIVER_NAMES[driver], 
                 linewidth=3, alpha=0.9)
        
        # Add start/end markers
        plt.scatter(subset['mds_x'].iloc[0], subset['mds_y'].iloc[0], 
                   c=colors[driver], marker='o', s=100, zorder=5)
        plt.scatter(subset['mds_x'].iloc[-1], subset['mds_y'].iloc[-1], 
                   c=colors[driver], marker='x', s=100, zorder=5)
    
    plt.title("Average Driving Style Comparison: Full Race (All Laps)")
    plt.xlabel("Similarity Dim 1")
    plt.ylabel("Similarity Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/driver_average_styles.png", dpi=300)
    print("Saved plot to output/driver_average_styles.png")

if __name__ == "__main__":
    analyze_full_race_styles()
