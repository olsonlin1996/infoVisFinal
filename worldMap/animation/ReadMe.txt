fetch_2024_races.py：執行後會抓取2024所有正賽車手的json檔
driver_json：存取2024年21個國家20位車手的json檔
fetch_2024_rankings.py：取得fastf1 的 laps 內的 Position 資料（每圈起點時間），會取 LapStartTime（或退回 Time）為時間軸，計算相對首圈的毫秒時間，輸出 {t, pos, lap}。
driver_positions：存有上述的資料