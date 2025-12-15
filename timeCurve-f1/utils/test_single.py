import requests

def log(msg):
    with open("test_single.log", "a") as f:
        f.write(msg + "\n")
    print(msg)

def main():
    with open("test_single.log", "w") as f: f.write("")
    
    url = "https://api.openf1.org/v1/car_data"
    
    # Test date>=  (EXPECT FAIL based on history?)
    params_ge = {
        'session_key': 9496,
        'driver_number': 1,
        'date>=': "2024-04-07T05:06:00",
        'date<': "2024-04-07T05:06:10"
    }
    log("Testing date>= ...")
    try:
        r = requests.get(url, params=params_ge)
        log(f"Status: {r.status_code}")
    except Exception as e:
        log(f"Exc: {e}")

    # Test date> (EXPECT PASS)
    params_gt = {
        'session_key': 9496,
        'driver_number': 1,
        'date>': "2024-04-07T05:06:00",
        'date<': "2024-04-07T05:06:10"
    }
    log("Testing date> ...")
    try:
        r = requests.get(url, params=params_gt)
        log(f"Status: {r.status_code}")
    except Exception as e:
        log(f"Exc: {e}")

if __name__ == "__main__":
    main()
