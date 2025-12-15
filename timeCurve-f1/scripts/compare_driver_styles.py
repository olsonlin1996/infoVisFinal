
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
        df_loc = pd.read_csv(f'data/loc_{driver}.csv')
        df_laps = pd.read_csv(f'data/laps_{driver}.csv')
    except FileNotFoundError:
        print(f"  Missing files for {driver}")
        return None, None

    # Time format conversion
    df_car['date'] = pd.to_datetime(df_car['date'], format='mixed')
    df_loc['date'] = pd.to_datetime(df_loc['date'], format='mixed')
    df_laps['date_start'] = pd.to_datetime(df_laps['date_start'], format='mixed')
    
    # Merge car and loc
    df_car.sort_values('date', inplace=True)
    df_loc.sort_values('date', inplace=True)
    
    merged = pd.merge_asof(
        df_car, 
        df_loc[['date', 'x', 'y', 'z']], 
        on='date', 
        direction='nearest',
        tolerance=pd.Timedelta('200ms')
    )
    merged.dropna(subset=['x', 'y'], inplace=True)
    merged['driver'] = driver
    
    return merged, df_laps

def filter_by_location(df, x_range, y_range):
    mask = (
        (df['x'] >= x_range[0]) & (df['x'] <= x_range[1]) &
        (df['y'] >= y_range[0]) & (df['y'] <= y_range[1])
    )
    return df[mask].copy()

def analyze_average_styles():
    # 1. Determine ROI from reference driver (Verstappen)
    ref_data, ref_laps = load_data(1)
    if ref_data is None:
        return
    
    # Use Lap 2, 15s-45s to determine ROI
    ref_lap = ref_laps[ref_laps['lap_number'] == 2].iloc[0]
    t_start = pd.to_datetime(ref_lap['date_start']) + pd.Timedelta(seconds=15)
    t_end = pd.to_datetime(ref_lap['date_start']) + pd.Timedelta(seconds=45)
    
    ref_segment = ref_data[
        (ref_data['date'] >= t_start) & (ref_data['date'] <= t_end)
    ]
    
    min_x, max_x = ref_segment['x'].min(), ref_segment['x'].max()
    min_y, max_y = ref_segment['y'].min(), ref_segment['y'].max()
    
    print(f"Auto-detected ROI from Lap 2: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
    
    # Add buffer
    input_box = (min_x - 2500, max_x + 2500, min_y - 2500, max_y + 2500)
    
    # 2. Extract all laps for each driver and compute average
    driver_averages = []
    FIXED_POINTS = 30
    COMMON_STEPS = 100
    
    for driver in TARGET_DRIVERS:
        full_data, laps_data = load_data(driver)
        if full_data is None:
            continue
        
        lap_segments = []
        
        # Extract all laps
        for _, lap in laps_data.iterrows():
            lap_num = int(lap['lap_number'])
            l_start = lap['date_start']
            
            if pd.isna(lap['lap_duration']):
                continue
            
            l_end = l_start + pd.Timedelta(seconds=lap['lap_duration'])
            lap_df = full_data[(full_data['date'] >= l_start) & (full_data['date'] <= l_end)]
            
            # Spatial Filter
            roi_df = filter_by_location(lap_df, (input_box[0], input_box[1]), (input_box[2], input_box[3]))
            
            if len(roi_df) > 5:
                # Resample to fixed points
                t_orig = np.linspace(0, 1, len(roi_df))
                t_new = np.linspace(0, 1, FIXED_POINTS)
                
                resampled_data = {
                    'lap': [lap_num] * FIXED_POINTS,
                }
                
                for feat in FEATURES:
                    resampled_data[feat] = np.interp(t_new, t_orig, roi_df[feat])
                
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
    
    # 3. Combine and project to MDS
    combined_avg = pd.concat(driver_averages, ignore_index=True)
    
    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(combined_avg[FEATURES])
    
    # MDS
    print("Running MDS on average trajectories...")
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
    
    # 4. Save
    combined_avg.to_csv('output/driver_average_styles.csv', index=False)
    print("Saved data to output/driver_average_styles.csv")
    
    # 5. Plot
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
    
    plt.title("Average Driving Style Comparison: S-Curves (All Laps)")
    plt.xlabel("Similarity Dim 1")
    plt.ylabel("Similarity Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/driver_average_styles.png", dpi=300)
    print("Saved plot to output/driver_average_styles.png")

if __name__ == "__main__":
    analyze_average_styles()
