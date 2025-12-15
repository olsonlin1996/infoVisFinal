
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def generate_time_curve(driver_number):
    """
    Generates a Time Curve for a given driver.
    """
    # 1. Data Abstraction
    print(f"Loading data for driver {driver_number}...")
    try:
        car_data_path = f'data/car_data_{driver_number}.csv'
        car_data = pd.read_csv(car_data_path)
        # Convert date to datetime if needed for sorting, though index is usually chronological
        car_data['date'] = pd.to_datetime(car_data['date'])
        car_data.sort_values('date', inplace=True)
    except FileNotFoundError:
        print(f"Error: Data file not found for driver {driver_number}.")
        return

    # Feature Selection
    features = ['speed', 'throttle', 'brake', 'n_gear', 'rpm', 'drs']
    raw_data = car_data[features].dropna()
    
    # --- OPTIMIZATION: Downsample ---
    # F1 telemetry is high freq. For MDS (O(N^3)), we need N < 3000 roughly.
    # If we have ~25k points, taking every 10th point gives 2.5k points.
    SAMPLE_RATE = 10 
    sampled_data = raw_data.iloc[::SAMPLE_RATE].copy()
    sampled_times = car_data['date'].iloc[::SAMPLE_RATE].values
    
    print(f"Data downsampled from {len(raw_data)} to {len(sampled_data)} points (Step={SAMPLE_RATE}).")

    # --- NORMALIZATION ---
    # Crucial because RPM (12000) >>> Brake (1). Euclidean distance would be dominated by RPM.
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(sampled_data)
    
    # 2. Distance Matrix
    print("Constructing distance matrix...")
    dist_matrix = pairwise_distances(normalized_data, metric='euclidean')
    print(f"Distance matrix shape: {dist_matrix.shape}")

    # 3. MDS Projection
    print("Running MDS... (This may take a moment)")
    # n_jobs=-1 uses all CPU cores. max_iter reduced slightly for speed if needed.
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1)
    pos = mds.fit_transform(dist_matrix)
    print("MDS completed.")

    # 4. Refinement: Rotate to align time mostly left-to-right
    # Simple heuristic: fit a PCA, align PC1 to x-axis.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pos_rotated = pca.fit_transform(pos)
    
    # Ensure start is on the left (if x[0] > x[end], flip x)
    if pos_rotated[0, 0] > pos_rotated[-1, 0]:
        pos_rotated[:, 0] *= -1

    # Save Coordinates
    result_df = pd.DataFrame(pos_rotated, columns=['x', 'y'])
    # Add normalized time (0.0 to 1.0) for coloring
    time_progress = np.linspace(0, 1, len(result_df))
    result_df['progress'] = time_progress
    
    csv_path = f'output/time_curve_{driver_number}.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"Coordinates saved to {csv_path}")

    # 5 & 6. Visualization (Curve Drawing & Visual Encoding)
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    # Create the curve as a collection of segments to color them by time
    points = np.array([result_df['x'], result_df['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time_progress.min(), time_progress.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    
    # Set the values used for colormapping
    lc.set_array(time_progress)
    lc.set_linewidth(2) # Todo: link to speed?
    
    ax = plt.gca()
    ax.add_collection(lc)
    ax.set_xlim(result_df['x'].min() - 0.1, result_df['x'].max() + 0.1)
    ax.set_ylim(result_df['y'].min() - 0.1, result_df['y'].max() + 0.1)
    
    # Add start/end markers
    ax.scatter(result_df['x'].iloc[0], result_df['y'].iloc[0], c='green', s=100, label='Start')
    ax.scatter(result_df['x'].iloc[-1], result_df['y'].iloc[-1], c='red', s=100, label='End')
    
    plt.title(f"F1 Time Curve: {driver_number} (2024 Japan)")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    img_path = f'output/time_curve_{driver_number}.png'
    plt.savefig(img_path, dpi=150)
    print(f"Plot saved to {img_path}")
    # plt.show()


if __name__ == '__main__':
    drivers = [1, 11, 22]
    for d in drivers:
        generate_time_curve(d)