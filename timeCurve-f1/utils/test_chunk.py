import requests
import json

def get(url):
    print(f"GET {url}")
    try:
        r = requests.get(url)
        print(f"Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Response: {r.text[:500]}")
        else:
            data = r.json()
            print(f"Items: {len(data)}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Test valid sub-range
    # Max lap 2 start
    start = "2024-04-07T05:05:53.290000"
    end = "2024-04-07T05:07:30.000000"
    
    url = f"https://api.openf1.org/v1/car_data?session_key=9496&driver_number=1&date_start={start}&date_end={end}"
    get(url)

if __name__ == "__main__":
    main()
