
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from matplotlib.collections import LineCollection

# --- CONFIGURATION ---
# Approximate Scale for Suzuka S-Curves (Need to verify with plot first)
# For now, we will pick a segment based on Lap 2 of Max, then use that ROI.
TARGET_DRIVERS = [1, 11, 22]
FEATURES = ['throttle', 'brake', 'n_gear']

def load_and_merge(driver):
    print(f"Loading data for Driver {driver}...")
    try:
        df_car = pd.read_csv(f'data/car_data_{driver}.csv')
        df_loc = pd.read_csv(f'data/loc_{driver}.csv')
    except FileNotFoundError:
        print(f"  Missing files for {driver}")
        return None


    # Convert dates with mixed format support (some have microseconds, some don't)
    df_car['date'] = pd.to_datetime(df_car['date'], format='mixed')
    df_loc['date'] = pd.to_datetime(df_loc['date'], format='mixed')
    
    # Sort
    df_car.sort_values('date', inplace=True)
    df_loc.sort_values('date', inplace=True)
    
    # Merge using merge_asof (nearest time match)
    # Tolerance: 200ms (Telemtry is ~3-4Hz, Location is similar)
    merged = pd.merge_asof(
        df_car, 
        df_loc[['date', 'x', 'y', 'z']], 
        on='date', 
        direction='nearest',
        tolerance=pd.Timedelta('200ms')
    )
    
    # Drop rows where location wasn't found
    merged.dropna(subset=['x', 'y'], inplace=True)
    merged['driver'] = driver
    return merged

def determine_roi(ref_driver=1):
    """
    Plots the track map for the reference driver and asks/determines coordinates.
    Since we are automated, we will use a specific Lap's segment index.
    Suzuka Sector 1 is roughly index 0 to 20% of the lap?
    The S-Curves are T3-T7.
    
    Let's just filter by a known geographic box if poss, or just segments.
    For this 'Demo', we will take a slice of x,y space.
    """
    df = load_and_merge(ref_driver)
    if df is None: return None
    
    # Take a look at x, y range
    # print(df['x'].min(), df['x'].max())
    # print(df['y'].min(), df['y'].max())
    
    # Heuristic: Suzuka S-curves are usually the winding part after the first straight (which is bottom-right to top-left).
    # Let's crudely slice by index for now to demonstrate mechanism, 
    # OR define a box. 
    # Let's try to define a box based on the first lap's shape.
    
    # HARDCODED ROI for 'S-Curves' approximation (Needs visual tuning)
    # Assuming standard OpenF1 coords.
    # We will return the whole df to plot it first if this was interactive.
    # Here we will define a broad box.
    return -10000, 10000, -10000, 10000 # Dummy

def analyze_corner():
    # 1. Load All Data
    all_data = []
    for d in TARGET_DRIVERS:
        df = load_and_merge(d)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("No data loaded.")
        return

    full_df = pd.concat(all_data)
    
    # 2. Filter for ROI (The "S-Curves")
    # To find the ROI, let's just create a "Track Map" first to see the shape.
    plt.figure(figsize=(10, 10))
    for d in TARGET_DRIVERS:
        subset = full_df[full_df['driver'] == d]
        plt.plot(subset['x'], subset['y'], label=f'Driver {d}', alpha=0.5, linewidth=1)
    
    plt.title("Track Map - Identify S-Curves")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("output/track_map_full.png")
    print("Full track map saved to output/track_map_full.png. Please identify ROI.")
    
    # --- AUTOMATIC ROI SELECTION (Approximation) ---
    # Suzuka S-Curves are typically in the geometric 'top-left' quadrant relative to the finish line?
    # Let's simplify: We will take a time-slice of Lap 2 (clean lap) from [15s to 45s] into the lap.
    # This usually covers T1 to Degner 1.
    
    print("Slicing data by Lap Time window (15s - 45s into lap)...")
    
    filtered_segments = []
    
    # Load Laps to get Start Time
    for d in TARGET_DRIVERS:
        laps_df = pd.read_csv(f'data/laps_{d}.csv')
        # Pick Lap 5 (usually stable)
        ref_lap = laps_df[laps_df['lap_number'] == 5].iloc[0]
        start_time = pd.to_datetime(ref_lap['date_start'])
        
        # Window: Start + 15s to Start + 45s
        t_start = start_time + pd.Timedelta(seconds=15)
        t_end = start_time + pd.Timedelta(seconds=45)
        
        # Get Telemetry in this window
        driver_telemetry = full_df[
            (full_df['driver'] == d) & 
            (full_df['date'] >= t_start) & 
            (full_df['date'] <= t_end)
        ].copy()
        
        if not driver_telemetry.empty:
            filtered_segments.append(driver_telemetry)
            print(f"  Driver {d}: {len(driver_telemetry)} points in S-Curve window.")

    if not filtered_segments:
        print("No data in window.")
        return

    combined_roi = pd.concat(filtered_segments)
    
    # Verify ROI visually
    plt.figure(figsize=(8, 8))
    for d in TARGET_DRIVERS:
        subset = combined_roi[combined_roi['driver'] == d]
        plt.plot(subset['x'], subset['y'], label=f'Driver {d}')
    plt.title("ROI Check: Segmented Curve")
    plt.axis('equal')
    plt.legend()
    plt.savefig("output/roi_check.png")
    
    # 3. Combined Time Curve Analysis
    # Downsample
    STEP = 2 # Less downsampling needed for short segment
    sampled_roi = combined_roi.iloc[::STEP].copy()
    
    # Normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(sampled_roi[FEATURES])
    
    # MDS
    print("Running Combined MDS...")
    dist = pairwise_distances(X, metric='euclidean')
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1)
    pos = mds.fit_transform(dist)
    
    # Rotate
    pca = PCA(n_components=2)
    pos = pca.fit_transform(pos)
    if pos[0,0] > pos[-1,0]: pos[:,0] *= -1
    
    sampled_roi['mds_x'] = pos[:, 0]
    sampled_roi['mds_y'] = pos[:, 1]
    
    # 4. Plot Comparison
    plt.figure(figsize=(12, 8))
    colors = {1: 'blue', 11: 'green', 22: 'red'}
    names = {1: 'Verstappen', 11: 'Perez', 22: 'Tsunoda'}
    
    for d in TARGET_DRIVERS:
        subset = sampled_roi[sampled_roi['driver'] == d]
        # Draw curve
        plt.plot(subset['mds_x'], subset['mds_y'], 
                 c=colors[d], label=names[d], linewidth=2, alpha=0.8)
        
        # Add Markers for key indices to show direction
        plt.scatter(subset['mds_x'].iloc[0], subset['mds_y'].iloc[0], c=colors[d], marker='o', s=50) # Start
        plt.scatter(subset['mds_x'].iloc[-1], subset['mds_y'].iloc[-1], c=colors[d], marker='x', s=50) # End


    plt.title("Driver Comparison (S-Curves): Shared Similarity Space")
    plt.xlabel("Similarity Dim 1")
    plt.ylabel("Similarity Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/comparison_scurve.png")
    print("Comparison saved to output/comparison_scurve.png")

    # Save Data for D3.js and Analysis
    sampled_roi.to_csv("output/comparison_scurve_data.csv", index=False)
    print("Data saved to output/comparison_scurve_data.csv")


if __name__ == "__main__":
    try:
        analyze_corner()
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("crash_log.txt", "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
