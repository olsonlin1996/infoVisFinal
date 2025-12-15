import requests
import pandas as pd
import os
import time
from datetime import timedelta


BASE_URL = "https://api.openf1.org/v1"

# Logging setup
LOG_FILE = "fetch_log_final_v2.txt"
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def get_json(url):
    log(f"Fetching {url}...")
    try:
        response = requests.get(url)
        # Handle 500 or 422 by returning empty or retrying?
        # For initial metadata, we expect 200.
        if response.status_code != 200:
            log(f"Error {response.status_code}: {response.text[:200]}")
            return []
        data = response.json()
        log(f"  -> Got {len(data)} items.")
        return data
    except Exception as e:
        log(f"Error fetching {url}: {e}")
        return []

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Starting fetch V3 (Time Chunks)...\n")

    # 1. Get meeting key for 2024 Japan
    meetings = get_json(f"{BASE_URL}/meetings?year=2024&country_name=Japan")
    if not meetings:
        log("No meetings found for 2024 Japan.")
        return
    meeting_key = meetings[0]['meeting_key']
    log(f"Meeting Key: {meeting_key}")

    # 2. Get session key for Race
    sessions = get_json(f"{BASE_URL}/sessions?meeting_key={meeting_key}&session_name=Race")
    if not sessions:
        log("No race session found.")
        return
    session_key = sessions[0]['session_key']
    log(f"Session Key: {session_key}")

    # 3. Target Drivers
    target_drivers = [1, 11, 22]
    os.makedirs('data', exist_ok=True)

    for driver_number in target_drivers:
        log(f"\n--- Processing Driver: {driver_number} ---")
        
        # 4. Laps
        laps_url = f"{BASE_URL}/laps?session_key={session_key}&driver_number={driver_number}"
        laps_data = get_json(laps_url)
        if not laps_data:
            log(f"No laps data for driver {driver_number}")
            continue

        df_laps = pd.DataFrame(laps_data)
        df_laps['date_start'] = pd.to_datetime(df_laps['date_start'])
        df_laps.to_csv(f"data/laps_{driver_number}.csv", index=False)
        log(f"Saved data/laps_{driver_number}.csv ({len(df_laps)} records)")
        
        # Determine Time Range from Laps
        start_time = df_laps['date_start'].min().floor('1min')
        # Add max duration to last lap start to get end time
        last_lap = df_laps.loc[df_laps['date_start'].idxmax()]
        duration = last_lap['lap_duration'] if not pd.isna(last_lap['lap_duration']) else 120
        end_time = (last_lap['date_start'] + pd.Timedelta(seconds=duration)).ceil('1min')
        
        log(f"Time Range: {start_time} to {end_time}")

        # Fetch Function
        def fetch_in_chunks(endpoint_name):
            all_records = []
            current = start_time
            chunk_size = pd.Timedelta(minutes=1) # 1 minute chunks
            
            while current < end_time:
                next_chunk = current + chunk_size
                

                # Format (Offset start by -1s and use date> to avoid date>= crash)
                # And use date< for end (exclusive).
                # This ensures we cover [start, start+1) which date> start misses, 
                # by doing date> start-1. 
                # Overlaps are handled by drop_duplicates.
                
                s_str = (current - pd.Timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%S')
                e_str = next_chunk.strftime('%Y-%m-%dT%H:%M:%S')
                
                params = {
                    'session_key': session_key,
                    'driver_number': driver_number,
                    'date>': s_str,
                    'date<': e_str
                }
                
                try:
                    r = requests.get(f"{BASE_URL}/{endpoint_name}", params=params)
                    if r.status_code == 200:
                        data = r.json()
                        if data: all_records.extend(data)
                    else:
                        log(f"  {endpoint_name} {s_str} failed: {r.status_code}")
                except Exception as e:
                    log(f"  {endpoint_name} {s_str} error: {e}")
                
                current = next_chunk
                time.sleep(0.1)
                
            return all_records

        # 5. Car Data
        log("Fetching car_data...")
        car_records = fetch_in_chunks("car_data")
        if car_records:
            df_car = pd.DataFrame(car_records)
            df_car.drop_duplicates(subset=['date'], inplace=True)
            df_car.to_csv(f"data/car_data_{driver_number}.csv", index=False)
            log(f"Saved data/car_data_{driver_number}.csv ({len(df_car)} records)")
        else:
            log("No car data found.")

        # 6. Location
        log("Fetching location...")
        loc_records = fetch_in_chunks("location")
        if loc_records:
            df_loc = pd.DataFrame(loc_records)
            df_loc.drop_duplicates(subset=['date'], inplace=True)
            df_loc.to_csv(f"data/loc_{driver_number}.csv", index=False)
            log(f"Saved data/loc_{driver_number}.csv ({len(df_loc)} records)")
        else:
            log("No location data found.")

    log("\nDone.")

if __name__ == "__main__":
    main()
