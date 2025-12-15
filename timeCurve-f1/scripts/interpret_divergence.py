
import pandas as pd
import numpy as np

def analyze():
    try:
        df = pd.read_csv('output/comparison_scurve_data.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter drivers
    df_max = df[df['driver'] == 1].copy().reset_index(drop=True)
    df_perez = df[df['driver'] == 11].copy().reset_index(drop=True)

    if df_max.empty or df_perez.empty:
        print("Data for one or both drivers is empty.")
        return

    # Align length
    min_len = min(len(df_max), len(df_perez))
    df_max = df_max.iloc[:min_len]
    df_perez = df_perez.iloc[:min_len]

    # Calculate Distance in MDS space
    diff_x = df_max['mds_x'] - df_perez['mds_x']
    diff_y = df_max['mds_y'] - df_perez['mds_y']
    dist = np.sqrt(diff_x**2 + diff_y**2)

    # Find Max Divergence
    max_idx = dist.idxmax()
    max_val = dist.max()

    with open('output/divergence_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"--- F1 S-Curve Divergence Analysis ---\n")
        f.write(f"Segments analyzed: {min_len} points\n")
        f.write(f"Maximum Divergence at Index: {max_idx}\n")
        f.write(f"Euclidean Distance in Similarity Space: {max_val:.4f}\n\n")
        
        f.write("Max Verstappen State:\n")
        f.write(df_max.iloc[max_idx][['speed', 'throttle', 'brake', 'n_gear', 'rpm']].to_string())
        f.write("\n\n")
        
        f.write("Sergio Perez State:\n")
        f.write(df_perez.iloc[max_idx][['speed', 'throttle', 'brake', 'n_gear', 'rpm']].to_string())
        f.write("\n")

    print("Analysis complete. Saved to output/divergence_report.txt")

if __name__ == "__main__":
    analyze()
