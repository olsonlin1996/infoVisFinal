"""
Fetch all 2024 race (R) telemetry and export per-driver JSON into
driver_json/{year}_{event_slug}/{code}.json plus drivers.json.

Usage:
  python fetch_2024_races.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fastf1
from fastf1 import events
import pandas as pd

OUT_ROOT = Path("driver_json")
YEAR = 2024
SESSION_TYPE = "R"  # Race


def slugify(name: str) -> str:
    """Convert event name to lowercase slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "race"


def build_points(session, driver_code: str):
    laps = session.laps.pick_drivers(driver_code)
    pos = laps.get_pos_data()
    if pos.empty:
        raise ValueError(f"No position data for driver {driver_code}")

    df = (
        pos.set_index("Date")[["X", "Y", "Z"]]
        .resample("100ms")
        .mean()
        .interpolate()
    )
    df["delta_x"] = df["X"].diff()
    df["delta_y"] = df["Y"].diff()
    df["distance"] = (df["delta_x"] ** 2 + df["delta_y"] ** 2) ** 0.5
    df = df[df["distance"] > 0.1].copy()
    if df.empty:
        raise ValueError(f"No movement after filtering for driver {driver_code}")

    t0 = df.index[0]
    points = []
    for ts, row in df.iterrows():
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            continue
        points.append(
            {
                "t": int((ts - t0).total_seconds() * 1000),
                "x": float(row["X"]),
                "y": float(row["Y"]),
            }
        )
    if not points:
        raise ValueError(f"No valid points for driver {driver_code}")
    return points


def export_race(event_row) -> None:
    round_no = int(event_row["RoundNumber"])
    event_name = event_row["EventName"]
    country = event_row.get("Country") or ""
    base_slug_source = country or event_name
    event_slug = slugify(base_slug_source)
    out_dir = OUT_ROOT / f"{YEAR}_{event_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Round {round_no}: {event_name} -> {out_dir} ===")
    session = fastf1.get_session(YEAR, round_no, SESSION_TYPE)
    session.load(telemetry=True, weather=False, messages=False)

    drivers_meta = []
    for drv in session.drivers:
        info = session.get_driver(drv)
        code = info["Abbreviation"]
        number = info["DriverNumber"]
        name = info["BroadcastName"]
        drivers_meta.append({"code": code, "number": number, "name": name})

        try:
            points = build_points(session, code)
            with open(out_dir / f"{code}.json", "w", encoding="utf-8") as f:
                json.dump({"points": points}, f, ensure_ascii=False)
            print(f"  Saved {code}.json ({len(points)} points)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Skip {code}: {exc}")

    with open(out_dir / "drivers.json", "w", encoding="utf-8") as f:
        json.dump({"drivers": drivers_meta}, f, ensure_ascii=False)
    print(f"  Saved drivers.json with {len(drivers_meta)} drivers\n")


def main():
    fastf1.Cache.enable_cache("cache")
    schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
    races = schedule[schedule["EventFormat"].notna()]  # filter out non-race/test

    for _, ev in races.iterrows():
        try:
            export_race(ev)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed {ev['EventName']}: {exc}")


if __name__ == "__main__":
    main()
