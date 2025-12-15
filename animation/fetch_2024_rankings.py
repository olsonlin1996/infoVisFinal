"""
Fetch 2024 race sessions from fastf1 and export per-driver live position
time series into driver_positions/{year}_{slug}/{code}.json.

Usage:
  python fetch_2024_rankings.py

Notes:
  - Requires `fastf1` installed and network access on first run
    (uses cache/ for subsequent runs).
  - Data source: session.timing_data (Position, optionally gaps/intervals).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import fastf1
import pandas as pd

YEAR = 2024
SESSION_TYPE = "R"  # Race
OUT_ROOT = Path("driver_positions")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "race"


def series_get(series: pd.Series, ts) -> Any:
    try:
        return series.loc[ts]
    except Exception:  # noqa: BLE001
        return None


def export_session(round_no: int, event_name: str) -> None:
    slug = f"{YEAR}_{slugify(event_name)}"
    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {event_name} (Round {round_no}) => {out_dir} ===")
    session = fastf1.get_session(YEAR, round_no, SESSION_TYPE)
    session.load(telemetry=False, weather=False, messages=False, laps=True)

    drivers_meta: List[Dict[str, Any]] = []
    for drv in session.drivers:
        info = session.get_driver(drv)
        code = info["Abbreviation"]
        number = int(info["DriverNumber"])
        name = info["BroadcastName"]
        drivers_meta.append({"code": code, "number": number, "name": name})

        laps = session.laps.pick_driver(drv)
        if laps.empty or "Position" not in laps:
            print(f"  [WARN] {code} Position not available; skip")
            continue

        # Prefer LapStartTime if available; otherwise fallback to 'Time'
        time_col = "LapStartTime" if "LapStartTime" in laps else "Time"
        if time_col not in laps:
            print(f"  [WARN] {code} has no lap time column; skip")
            continue

        laps = laps.dropna(subset=["Position", time_col])
        if laps.empty:
            print(f"  [WARN] {code} no usable laps; skip")
            continue

        t0 = laps.iloc[0][time_col]
        records = []
        for _, row in laps.iterrows():
            pos = row["Position"]
            lap_ts = row[time_col]
            if pd.isna(pos) or pd.isna(lap_ts):
                continue
            records.append(
                {
                    "t": int((lap_ts - t0).total_seconds() * 1000),
                    "pos": int(pos),
                    "lap": int(row["LapNumber"]) if "LapNumber" in row and not pd.isna(row["LapNumber"]) else None,
                }
            )

        out_path = out_dir / f"{code}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"  Saved {code}.json ({len(records)} points)")

    with (out_dir / "drivers.json").open("w", encoding="utf-8") as f:
        json.dump({"drivers": drivers_meta}, f, ensure_ascii=False)
    print(f"  Saved drivers.json ({len(drivers_meta)} drivers)\\n")


def main() -> None:
    fastf1.Cache.enable_cache("cache")
    schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
    races = schedule[schedule["EventFormat"].notna()]
    for _, ev in races.iterrows():
        export_session(int(ev["RoundNumber"]), ev["EventName"])


if __name__ == "__main__":
    main()
