import requests

def get(url):
    print(f"GET {url}")
    r = requests.get(url)
    print(f"Status: {r.status_code}")
    if r.status_code != 200:
        print(f"Response: {r.text[:500]}")
    else:
        data = r.json()
        print(f"Items: {len(data)}")
        if len(data) > 0:
            print(f"Sample: {data[0]}")

def main():
    # 1. Verify Drivers
    get("https://api.openf1.org/v1/drivers?session_key=9496")
    
    # 2. Try car_data with very limited scope
    # Maybe limit=1 if supported? Or small time window?
    # OpenF1 supports specific filters.
    print("\n--- Testing car_data ---")
    get("https://api.openf1.org/v1/car_data?session_key=9496&driver_number=1&n_gear=2") # Random filter to reduce size?

    # 3. Try location
    print("\n--- Testing location ---")
    get("https://api.openf1.org/v1/location?session_key=9496&driver_number=1&date>2024-04-07T06:00:00&date<2024-04-07T06:01:00")

if __name__ == "__main__":
    main()
