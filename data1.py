import fastf1
import pandas as pd
import json
import os

if not os.path.exists("cache"):
    os.makedirs("cache")
fastf1.Cache.enable_cache("cache")


def get_lightweight_season_data(year):
    print(f"正在獲取 {year} 賽季資料 (輕量版)...")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    season_data = []

    for i, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        gp_name = event["EventName"]

        # 排除還沒發生的比賽 (例如抓取時如果還沒比完)
        # 如果你想保留未來賽程但沒數據，可以加 try-except

        print(f"[{round_num}/{len(schedule)}] 處理: {gp_name} ...")

        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(telemetry=False, laps=False, weather=True, messages=False)

            # --- 1. 處理前三名 ---
            cols = ["Abbreviation", "TeamName", "GridPosition", "ClassifiedPosition"]
            # 注意：有些比賽可能沒成績，這裡做個保護
            if hasattr(session, "results") and not session.results.empty:
                top3_df = session.results.iloc[:3][cols].copy()
                podium_data = top3_df.to_dict(orient="records")
            else:
                podium_data = []

            # --- 2. 處理天氣 (重點：只算平均值) ---
            weather_df = session.weather_data

            # 計算平均氣溫與賽道溫 (取小數點後 1 位)
            avg_air = round(weather_df["AirTemp"].mean(), 1)
            avg_track = round(weather_df["TrackTemp"].mean(), 1)

            # 判斷是否有下雨 (只要任何一分鐘 Rain=True 就算有雨)
            # Rainfall 在 fastf1 裡是 Boolean 欄位
            has_rain = weather_df["Rainfall"].any()

            weather_summary = {
                "AirTemp": avg_air,
                "TrackTemp": avg_track,
                "IsRainy": bool(has_rain),  # 轉成標準 True/False
            }

            race_entry = {
                "Round": int(round_num),
                "RaceName": gp_name,
                "Country": event["Country"],
                "Date": str(event["EventDate"]),
                "Podium": podium_data,
                "Weather": weather_summary,  # 這裡不再是幾百筆陣列，而是一個小字典
            }

            season_data.append(race_entry)

        except Exception as e:
            print(f"  -> 跳過 {gp_name} (無數據或發生錯誤)")
            # 即使沒數據，也可以存一個只有基本資訊的項目，避免前端報錯
            season_data.append(
                {
                    "Round": int(round_num),
                    "RaceName": gp_name,
                    "Country": event["Country"],
                    "Date": str(event["EventDate"]),
                    "Podium": [],
                    "Weather": None,
                }
            )

    return season_data


if __name__ == "__main__":
    data = get_lightweight_season_data(2024)
    filename = "f1_2024_weather_and_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("完成！檔案已瘦身。")
