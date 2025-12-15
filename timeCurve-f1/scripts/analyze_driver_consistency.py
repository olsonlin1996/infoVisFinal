
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import sys

# --- CONFIGURATION ---
# Accept driver number from command line, default to Verstappen
TARGET_DRIVER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DRIVER_NAMES = {1: 'Verstappen', 11: 'Perez', 22: 'Tsunoda'}
FEATURES = ['throttle', 'brake', 'n_gear'] # Keep consistent with index
# Explicitly specific laps to highlight, or None for all
# HIGHLIGHT_LAPS = [2, 18, 50] 
HIGHLIGHT_LAPS = None 

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
    
    return merged, df_laps

def filter_by_location(df, x_range, y_range):
    """
    Filter data points that fall within a bounding box.
    """
    mask = (
        (df['x'] >= x_range[0]) & (df['x'] <= x_range[1]) &
        (df['y'] >= y_range[0]) & (df['y'] <= y_range[1])
    )
    return df[mask].copy()

def analyze_driver_laps():
    full_data, laps_data = load_data(TARGET_DRIVER)
    if full_data is None: return

    # 1. Define ROI (S-Curves)
    # Based on previous analysis or comparison_scurve_data.csv inspection
    # Let's peek at the comparison data to determine bounds if possible, 
    # or just use the domain we saw earlier.
    # Looking at comparison_scurve_data.csv (from prev steps), X seems to be around -6000 to +4000
    # Y seems to be around -6000 to -1000.
    # To be safe, let's use the track map logic again or strictly filter by 'Sector 1' logic?
    # Spatial is safer. Max's data in the csv shows:
    # x approx: -6144 to 1232 (S-Curves segment we extracted before)
    # y approx: -2870 to 4104
    # Wait, the indices in analyze_corner were time-based.
    # Let's define a box that captures T3-T6.
    # Rough Estimate from standard Suzuka map data:
    # X: [-6500, -1000], Y: [-3000, 1000] (Just a guess, likely needs tuning)
    
    # Better approach: 
    # Use the logic from analyze_corner (Lap 2, 15s-45s) to find the bounding box dynamically first!
    ref_lap = laps_data[laps_data['lap_number'] == 2].iloc[0]
    t_start = pd.to_datetime(ref_lap['date_start']) + pd.Timedelta(seconds=15)
    t_end = pd.to_datetime(ref_lap['date_start']) + pd.Timedelta(seconds=45)
    
    ref_segment = full_data[
        (full_data['date'] >= t_start) & (full_data['date'] <= t_end)
    ]
    
    min_x, max_x = ref_segment['x'].min(), ref_segment['x'].max()
    min_y, max_y = ref_segment['y'].min(), ref_segment['y'].max()
    
    print(f"Auto-detected ROI from Lap 2: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
    
    # Buffers
    # Increase buffer to 2500 to catch all laps despite GPS drift or line changes
    input_box = (min_x - 2500, max_x + 2500, min_y - 2500, max_y + 2500)

    # 2. Extract Segments for Every Lap
    lap_segments = []
    FIXED_POINTS = 30  # Downsample to speed up MDS (N^2 complexity)

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
            # Resample to fixed number of points using interpolation
            # Create a localized 0-1 index
            t_orig = np.linspace(0, 1, len(roi_df))
            t_new = np.linspace(0, 1, FIXED_POINTS)
            
            resampled_data = {
                'lap': [lap_num] * FIXED_POINTS,
                # Interpolate Features and Coords
                # We need x, y for plotting context if needed (not strictly used by MDS but good to have)
                'x': np.interp(t_new, t_orig, roi_df['x']),
                'y': np.interp(t_new, t_orig, roi_df['y'])
            }
            
            # Interpolate Features
            for feat in FEATURES:
                resampled_data[feat] = np.interp(t_new, t_orig, roi_df[feat])
                
            resampled_df = pd.DataFrame(resampled_data)
            lap_segments.append(resampled_df)
    
    print(f"Extracted {len(lap_segments)} valid lap segments (resampled to {FIXED_POINTS} points each).")
    
    if not lap_segments:
        print("No data found in ROI.")
        return

    combined_laps = pd.concat(lap_segments)
    
    # 3. MDS Projection
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(combined_laps[FEATURES])
    
    # Downsample for speed? (53 laps * ~50 points = 2500 points, MDS handles fine)
    # But for visual clarity, let's keep it dense.
    
    print("Running MDS on multi-lap data...")
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, n_jobs=-1)
    pos = mds.fit_transform(X_scaled)
    
    # Rotate with PCA
    pca = PCA(n_components=2)
    pos = pca.fit_transform(pos)
    if pos[0,0] > pos[-1,0]: pos[:,0] *= -1
    
    combined_laps['mds_x'] = pos[:, 0]
    combined_laps['mds_y'] = pos[:, 1]
    
    # --- Find Fastest Lap ---
    # Instead of averaging all laps, use the fastest lap as reference
    
    print("Finding Fastest Lap...")
    
    # Find the lap with minimum duration
    fastest_lap_row = laps_data.loc[laps_data['lap_duration'].idxmin()]
    fastest_lap_num = int(fastest_lap_row['lap_number'])
    fastest_lap_time = fastest_lap_row['lap_duration']
    
    print(f"Fastest Lap: {fastest_lap_num} (Duration: {fastest_lap_time:.3f}s)")
    
    # Extract fastest lap data from combined_laps
    fastest_lap_data = combined_laps[combined_laps['lap'] == fastest_lap_num].copy()
    
    if len(fastest_lap_data) > 0:
        # Resample to common steps for consistent comparison
        COMMON_STEPS = 100
        t = np.linspace(0, 1, len(fastest_lap_data))
        
        fastest_x = np.interp(np.linspace(0, 1, COMMON_STEPS), t, fastest_lap_data['mds_x'])
        fastest_y = np.interp(np.linspace(0, 1, COMMON_STEPS), t, fastest_lap_data['mds_y'])
        
        # Create fastest lap reference dataframe
        fastest_df = pd.DataFrame({
            'lap': [-1] * COMMON_STEPS,  # Special ID for Fastest Lap Reference
            'mds_x': fastest_x,
            'mds_y': fastest_y,
            'driver': [TARGET_DRIVER] * COMMON_STEPS
        })
        
        # Append
        combined_laps = pd.concat([combined_laps, fastest_df], ignore_index=True)
    else:
        print(f"Warning: Fastest lap {fastest_lap_num} data not found in ROI.")
    
    
    
    # 4. Plotting
    plt.figure(figsize=(14, 10))
    
    # Color map: Laps (Sequential)
    # Use a colormap like 'viridis' or 'plasma' to show time progression
    # Early laps = Purple/Blue, Late laps = Yellow
    
    cmap = plt.get_cmap('magma_r') # Reverse magma: Dark=Late, Light=Early? Or Viridis.
    # Let's use 'viridis': Purple(Early) -> Yellow(Late)
    cmap = plt.get_cmap('viridis')
    
    norm = plt.Normalize(vmin=combined_laps['lap'].min(), vmax=combined_laps['lap'].max())
    
    # Plot each lap as a line
    for lap_idx in sorted(combined_laps['lap'].unique()):
        subset = combined_laps[combined_laps['lap'] == lap_idx]
        
        # Color based on lap number
        color = cmap(norm(lap_idx))
        
        # Line width: Thinner for background, thicker for highlights?
        # Standard:
        plt.plot(subset['mds_x'], subset['mds_y'], color=color, alpha=0.9, linewidth=1.5)
        
        # Optional: Add small dots
        # plt.scatter(subset['mds_x'], subset['mds_y'], color=color, s=5, alpha=0.5)

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Lap Number')
    
    plt.title(f"Driver {TARGET_DRIVER} ({DRIVER_NAMES[TARGET_DRIVER]}) Consistency Analysis: S-Curves (Lap 1-{combined_laps['lap'].max()})")
    plt.xlabel("Similarity Dim 1")
    plt.ylabel("Similarity Dim 2")
    plt.grid(True, alpha=0.3)
    
    # Save with driver-specific filenames
    driver_name = DRIVER_NAMES[TARGET_DRIVER]
    plt.savefig(f"output/driver_laps_consistency_{driver_name}.png", dpi=300)
    print(f"Saved plot to output/driver_laps_consistency_{driver_name}.png")
    
    # Save CSV for potential D3
    combined_laps.to_csv(f"output/driver_laps_data_{driver_name}.csv", index=False)
    print(f"Saved data to output/driver_laps_data_{driver_name}.csv")

if __name__ == "__main__":
    analyze_driver_laps()
