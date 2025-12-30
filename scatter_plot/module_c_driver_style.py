"""
module_c_driver_style.py

C 模組：駕駛風格散佈圖（Driver Style Map）
- 對指定場次、數位車手，計算「每圈」的 throttle / brake / gear 統計特徵
- 用 PCA 降到 2D，輸出 JSON 給前端畫散佈圖 + 雷達圖

依賴：
    pip install requests pandas scikit-learn
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


OPENF1_BASE = "https://api.openf1.org/v1"

logger = logging.getLogger(__name__)


# ========== 型別定義 ==========

@dataclass
class LapFeature:
    driver: str           # driver_number (string)
    lap_number: int
    embed_x: float
    embed_y: float

    avg_throttle: float
    throttle_std: float
    full_throttle_ratio: float

    brake_ratio: float
    brake_intensity: float

    avg_gear: float
    gear_changes: int

    phase: str            # "opening" / "middle" / "ending"
    is_focus: bool        # 是否是特別標記的圈（例如 Lap2 / mid / end）
    track_status: str     # 這圈的狀態摘要（GREEN/SC/VSC/YELLOW/RED/CHEQUERED...）


# ========== 通用小工具 ==========

def normalize_str(s: str) -> str:
    """簡單正規化字串：lower + 去掉空白與底線。"""
    return re.sub(r"[\s_]+", "", s or "").lower()


def _slugify_race(text: str) -> str:
    """
    把 'Japanese Grand Prix' / 'Las Vegas Grand Prix' 之類轉成簡單 slug：
    'japanesegp', 'lasvegas'
    再拿來和你在 driver_style_races.txt 寫的 'japan', 'vegas' 做比對。
    """
    if text is None:
        return ""
    s = str(text).lower()

    # 拿掉常見尾巴字眼
    s = s.replace("grand prix", "")
    s = s.replace("gp", "")

    # 只留英數字
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# ========== API helpers ==========

def get_openf1(endpoint: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    呼叫 openF1 endpoint，回傳 pandas DataFrame。
    endpoint 例如: "laps", "car_data", "sessions", "race_control", "meetings", "session_result", "drivers"
    """
    url = f"{OPENF1_BASE}/{endpoint}"
    logger.debug("[get_openf1] GET %s params=%s", url, params)
    resp = requests.get(url, params=params, timeout=30)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.error("[get_openf1] HTTPError: %s", e)
        logger.error("[get_openf1] response text (first 500 chars): %s", resp.text[:500])
        raise

    data = resp.json()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def fetch_session_start(session_key: int) -> Optional[pd.Timestamp]:
    """
    從 /sessions 取得這個 session 的 date_start（UTC aware）。
    """
    df = get_openf1("sessions", {"session_key": session_key})
    if df.empty:
        logger.warning("[fetch_session_start] no sessions for session_key=%s", session_key)
        return None

    if "date_start" not in df.columns:
        logger.warning(
            "[fetch_session_start] sessions has no date_start column; columns=%s",
            list(df.columns),
        )
        return None

    # 預期只有一列，但還是取最小 date_start
    df["date_start"] = pd.to_datetime(
        df["date_start"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["date_start"])
    if df.empty:
        logger.warning("[fetch_session_start] date_start all NaT for session_key=%s", session_key)
        return None

    date_start = df["date_start"].min()
    logger.info("[fetch_session_start] session_key=%s, date_start=%s", session_key, date_start)
    return date_start


def fetch_race_control(session_key: int) -> pd.DataFrame:
    """
    抓整場的 /race_control，並做基本清洗：
      - 轉成 datetime
      - 依時間排序
      - 處理 CHEQUERED + RED 同時出現的 timestamp
      - 移除第一個 CHEQUERED FLAG 之後才出現的 RED 事件（視為賽後 RED）
    """
    df = get_openf1("race_control", {"session_key": session_key})
    if df.empty:
        logger.warning("[fetch_race_control] no race_control rows for session_key=%s", session_key)
        return df

    if "date" not in df.columns:
        logger.warning(
            "[fetch_race_control] race_control has no date column; columns=%s",
            list(df.columns),
        )
        return pd.DataFrame()

    df["date"] = pd.to_datetime(
        df["date"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "flag" not in df.columns:
        df["flag"] = ""
    if "message" not in df.columns:
        df["message"] = ""

    # 1) 移除同一 timestamp 中，CHEQUERED + RED 的 RED（只保留 CHEQUERED）
    duplicated_times = df["date"].value_counts()
    duplicated_times = duplicated_times[duplicated_times > 1].index

    red_dup_mask = pd.Series(False, index=df.index)
    for t in duplicated_times:
        subset = df[df["date"] == t]
        has_cheq = subset["message"].astype(str).str.upper().str.contains("CHEQUERED FLAG").any()
        has_red = subset["flag"].astype(str).str.upper().eq("RED").any()
        if has_cheq and has_red:
            mask_red_same_time = (df["date"] == t) & df["flag"].astype(str).str.upper().eq("RED")
            red_dup_mask |= mask_red_same_time

    if red_dup_mask.any():
        dropped = df[red_dup_mask]
        logger.info(
            "[fetch_race_control] drop %d RED rows at same timestamp as CHEQUERED FLAG",
            len(dropped),
        )
        logger.debug("[fetch_race_control] dropped RED@CHEQUERED rows:\n%s", dropped)
        df = df.loc[~red_dup_mask].reset_index(drop=True)

    # 2) 找出第一個 CHEQUERED FLAG 之後的 RED，視為賽後 RED，不掛到任何一圈
    msg_col_u = df["message"].astype(str).str.upper()
    cheq_mask = msg_col_u.str.contains("CHEQUERED FLAG")
    if cheq_mask.any():
        t_cheq = df.loc[cheq_mask, "date"].min()
        red_post_mask = (df["date"] >= t_cheq) & df["flag"].astype(str).str.upper().eq("RED")
        if red_post_mask.any():
            dropped2 = df.loc[red_post_mask]
            logger.info(
                "[fetch_race_control] drop %d RED rows at/after first CHEQUERED FLAG",
                len(dropped2),
            )
            logger.debug("[fetch_race_control] dropped post-CHEQUERED RED rows:\n%s", dropped2)
            df = df.loc[~red_post_mask].reset_index(drop=True)

    return df


def fetch_race_start_time(session_key: int, rc_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    從 race_control 裡找出「第一次 GREEN」時間作為比賽開始時間。
    如果找不到，就回退到 session 的 date_start。
    """
    session_start = fetch_session_start(session_key)

    if rc_df.empty:
        logger.warning("[fetch_race_start_time] empty race_control, fallback to session_start=%s", session_start)
        return session_start

    if "message" not in rc_df.columns:
        logger.warning("[fetch_race_start_time] race_control has no message column, fallback to session_start=%s", session_start)
        return session_start

    msg_u = rc_df["message"].astype(str).str.upper()
    green_mask = msg_u.str.contains("GREEN")
    if green_mask.any():
        t_green = rc_df.loc[green_mask, "date"].min()
        logger.info("[fetch_race_start_time] use first GREEN as race_start: %s", t_green)
        return t_green

    logger.warning("[fetch_race_start_time] no GREEN found, fallback to session_start=%s", session_start)
    return session_start


def fetch_laps(
    session_key: int,
    driver_number: int,
    rc_df: pd.DataFrame,
    race_start: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """
    取得某車手在該 session 的所有 laps 資料，補上第一圈的 date_start（如果缺）。
    重要欄位（實際名稱以 openF1 為準）：
      - lap_number
      - date_start （這圈開始時間）
    """
    df = get_openf1("laps", {
        "session_key": session_key,
        "driver_number": driver_number,
    })
    if df.empty:
        raise ValueError(f"No laps found for session={session_key}, driver={driver_number}")

    if "date_start" not in df.columns:
        raise ValueError("laps endpoint has no date_start column")
    if "lap_number" not in df.columns:
        raise ValueError("laps endpoint has no lap_number column")

    df["date_start"] = pd.to_datetime(
        df["date_start"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )

    # 如果第一圈的 date_start 為 NaT，試著用 race_start 填補
    min_lap = df["lap_number"].min()
    mask_first_lap = df["lap_number"].eq(min_lap)
    if df.loc[mask_first_lap, "date_start"].isna().any():
        if race_start is None:
            # 再從 race_control 裡找一次 GREEN
            race_start = fetch_race_start_time(session_key, rc_df)
        if race_start is not None:
            df.loc[mask_first_lap, "date_start"] = race_start
            logger.info(
                "[fetch_laps] fill lap %s date_start with race_start=%s",
                min_lap,
                race_start,
            )
        else:
            logger.warning("[fetch_laps] lap %s has no date_start and race_start is None", min_lap)

    df = df.dropna(subset=["date_start"])
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True)
    df = df.sort_values("lap_number").reset_index(drop=True)

    # 計算每圈 duration（秒），最後一圈沒有 end_time 就設為 NaN
    df["lap_duration_s"] = np.nan
    starts = df["date_start"].tolist()
    for i in range(len(df) - 1):
        dt = (starts[i + 1] - starts[i]).total_seconds()
        df.at[i, "lap_duration_s"] = dt

    logger.info(
        "[fetch_laps] driver=%s, laps: min=%s, max=%s, count=%s",
        driver_number,
        df["lap_number"].min(),
        df["lap_number"].max(),
        len(df),
    )
    logger.debug("[fetch_laps] head:\n%s", df.head())

    return df


def build_lap_status_mapping(
    laps_df: pd.DataFrame,
    rc_df: pd.DataFrame,
) -> Dict[int, str]:
    """
    把 race_control 事件 map 到每一圈，輸出 {lap_number: track_status_summary}。
    track_status_summary 例如：
        "GREEN"
        "SC"
        "VSC"
        "YELLOW"
        "SC;YELLOW"
        "CHEQUERED"
        "RED_FLAG"
    """
    if rc_df.empty:
        # 全場都當作 GREEN
        return {int(row.lap_number): "GREEN" for row in laps_df.itertuples(index=False)}

    if "date" not in rc_df.columns or "message" not in rc_df.columns:
        logger.warning("[build_lap_status_mapping] race_control missing date/message, treat all laps as GREEN")
        return {int(row.lap_number): "GREEN" for row in laps_df.itertuples(index=False)}

    status_by_lap: Dict[int, str] = {}
    rc_df = rc_df.sort_values("date").reset_index(drop=True)

    lap_numbers = laps_df["lap_number"].tolist()
    lap_starts = laps_df["date_start"].tolist()

    # 為了給最後一圈找 end_time，先準備一個 end_time list
    lap_ends: List[Optional[pd.Timestamp]] = []
    for i in range(len(lap_numbers) - 1):
        lap_ends.append(lap_starts[i + 1])
    lap_ends.append(None)  # 最後一圈到賽後

    for lap_num, start, end in zip(lap_numbers, lap_starts, lap_ends):
        if end is not None:
            mask = (rc_df["date"] >= start) & (rc_df["date"] < end)
        else:
            mask = rc_df["date"] >= start

        rc_in_lap = rc_df.loc[mask]
        msgs_u = rc_in_lap["message"].astype(str).str.upper()

        tags: List[str] = []

        # 根據 message 判斷狀態
        if msgs_u.str.contains("SAFETY CAR").any() and not msgs_u.str.contains("VIRTUAL SAFETY CAR").any():
            tags.append("SC")
        if msgs_u.str.contains("VIRTUAL SAFETY CAR").any():
            tags.append("VSC")
        if msgs_u.str.contains("YELLOW").any():
            tags.append("YELLOW")
        if msgs_u.str.contains("RED FLAG").any():
            tags.append("RED_FLAG")
        if msgs_u.str.contains("CHEQUERED FLAG").any():
            tags.append("CHEQUERED")

        if not tags:
            status = "GREEN"
        else:
            # 去重保持順序
            seen = set()
            uniq_tags = []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    uniq_tags.append(t)
            status = ";".join(uniq_tags)

        status_by_lap[int(lap_num)] = status

    logger.debug("[build_lap_status_mapping] status_by_lap=%s", status_by_lap)
    return status_by_lap


# ========== Feature 計算 ==========

def compute_features_for_lap(samples: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    對一圈的 samples（含 throttle/brake/gear）計算統計特徵。
    """
    if samples.empty:
        return None

    throttles = samples["throttle"].to_numpy(dtype=float)
    brakes = samples["brake"].to_numpy(dtype=float)
    gears = samples["gear"].to_numpy(dtype=float)

    n = len(samples)
    if n == 0:
        return None

    # 1) throttle
    avg_throttle = float(throttles.mean())
    throttle_std = float(throttles.std(ddof=0))
    full_throttle_ratio = float((throttles >= 0.9).sum() / n)

    # 2) brake
    brake_on = brakes[brakes > 0.05]
    brake_ratio = float((brakes > 0.05).sum() / n)
    brake_intensity = float(brake_on.mean()) if len(brake_on) > 0 else 0.0

    # 3) gear
    avg_gear = float(gears.mean())
    gear_changes = int((gears[1:] != gears[:-1]).sum())

    return {
        "avg_throttle": avg_throttle,
        "throttle_std": throttle_std,
        "full_throttle_ratio": full_throttle_ratio,
        "brake_ratio": brake_ratio,
        "brake_intensity": brake_intensity,
        "avg_gear": avg_gear,
        "gear_changes": gear_changes,
    }


def fetch_car_data_raw(
    session_key: int,
    driver_number: int,
    start: pd.Timestamp,
    end: Optional[pd.Timestamp] = None,
    min_speed: int = 0,
) -> pd.DataFrame:
    """
    只抓「某個時間區間」內的 car_data，避免一次拿整場爆 422。

    - start: 這圈的開始時間 (datetime, tz-aware)
    - end:   下一圈的開始時間；最後一圈可以給 None，表示抓到 session 結束
    """
    params: Dict[str, Any] = {
        "session_key": session_key,
        "driver_number": driver_number,
        "date>": start.isoformat(),
    }
    if end is not None:
        params["date<"] = end.isoformat()

    if min_speed is not None and min_speed > 0:
        params["speed>"] = min_speed

    df = get_openf1("car_data", params)
    if df.empty:
        return df

    if "date" not in df.columns:
        raise ValueError("car_data endpoint has no date column")

    for col in ["throttle", "brake", "n_gear"]:
        if col not in df.columns:
            raise ValueError(f"car_data endpoint has no {col} column")

    df["date"] = pd.to_datetime(
        df["date"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["date"])

    df["throttle"] = df["throttle"].astype(float) / 100.0
    df["brake"] = df["brake"].astype(float) / 100.0
    df["gear"] = df["n_gear"].astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "throttle", "brake", "gear"]]


def should_drop_lap(
    lap_number: int,
    lap_duration_s: float,
    track_status: str,
    typical_duration_s: Optional[float],
    long_lap_factor: float = 1.6,
) -> Tuple[bool, str]:
    """
    根據狀態與圈時間判斷這圈要不要丟掉。
    回傳 (should_drop, reason)，reason 供 log 記錄。

    規則（依優先順序）：
      1. SC / VSC 圈直接丟
      2. CHEQUERED FLAG 那圈丟（終點後 in-lap）
      3. RED FLAG 圈丟
      4. 含 YELLOW 的圈丟
      5. 異常過長的圈丟（疑似紅旗暫停或 safety car 拖很久）
    """
    status_u = track_status.upper() if isinstance(track_status, str) else ""

    # 1) SC / VSC
    if "SC" in status_u and "VSC" not in status_u:
        return True, "SC_LAP"
    if "VSC" in status_u:
        return True, "VSC_LAP"

    # 2) 終點圈（CHEQUERED 優先於 RED）
    if "CHEQUERED" in status_u:
        return True, "CHEQUERED_FLAG_LAP"

    # 3) 紅旗圈
    if "RED_FLAG" in status_u or "RED" in status_u:
        return True, "RED_FLAG_LAP"

    # 4) 黃旗圈
    if "YELLOW" in status_u:
        return True, "YELLOW_FLAG_LAP"

    # 5) 過長圈（時間判斷）
    if (
        typical_duration_s is not None
        and lap_duration_s is not None
        and not np.isnan(lap_duration_s)
        and lap_duration_s > typical_duration_s * long_lap_factor
    ):
        return True, f"LONG_LAP_{lap_duration_s:.1f}s"

    return False, ""


def compute_lap_features_for_driver(
    session_key: int,
    driver_number: int,
    rc_df: pd.DataFrame,
    phase_split: Dict[str, List[int]],
) -> List[Dict[str, Any]]:
    """
    對單一車手計算所有圈的特徵，回傳 list[dict]
    phase_split: {"opening": [...], "middle": [...], "ending": [...]}

    - 先抓 laps + race_control，算出每圈的 track_status / duration
    - 用 should_drop_lap 決定要不要 drop
    - 對保留下來的圈抓單圈 car_data，算特徵
    """
    laps_df = fetch_laps(session_key, driver_number, rc_df, race_start=None)
    laps_df = laps_df.sort_values("lap_number").reset_index(drop=True)

    # 建立 lap_number -> track_status 映射
    status_by_lap = build_lap_status_mapping(laps_df, rc_df)

    # 預估「正常圈時間」：取去掉極端值後的中位數
    durations = laps_df["lap_duration_s"].dropna()
    if not durations.empty:
        # 去除明顯太短 / 太長的 outliers，粗略限縮在 [0.5*median, 1.5*median]
        med = durations.median()
        mask_norm = (durations >= med * 0.5) & (durations <= med * 1.5)
        typical_duration_s = durations[mask_norm].median()
    else:
        typical_duration_s = None

    logger.info(
        "[compute_lap_features_for_driver] driver=%s, typical_duration_s=%s",
        driver_number,
        typical_duration_s,
    )

    lap_numbers = laps_df["lap_number"].tolist()
    lap_starts = laps_df["date_start"].tolist()
    lap_durations = laps_df["lap_duration_s"].tolist()

    out: List[Dict[str, Any]] = []

    for i, lap_num in enumerate(lap_numbers):
        start = lap_starts[i]
        end = lap_starts[i + 1] if i + 1 < len(lap_numbers) else None
        duration_s = lap_durations[i]
        track_status = status_by_lap.get(int(lap_num), "GREEN")

        drop, reason = should_drop_lap(
            lap_number=int(lap_num),
            lap_duration_s=duration_s,
            track_status=track_status,
            typical_duration_s=typical_duration_s,
        )

        if drop:
            logger.info(
                "[compute_lap_features_for_driver] driver=%s lap=%s DROP (%s), status=%s, duration=%.1fs",
                driver_number,
                lap_num,
                reason,
                track_status,
                duration_s if duration_s is not None else -1.0,
            )
            continue

        logger.info(
            "[compute_lap_features_for_driver] driver=%s lap=%s KEEP, status=%s, duration=%.1fs",
            driver_number,
            lap_num,
            track_status,
            duration_s if duration_s is not None else -1.0,
        )

        # 單圈的 car_data
        car_df = fetch_car_data_raw(
            session_key=session_key,
            driver_number=driver_number,
            start=start,
            end=end,
            min_speed=0,      # 這裡不篩 min_speed，避免把整圈吃掉
        )

        logger.debug(
            "[compute_lap_features_for_driver] driver=%s lap=%s, car_data rows=%s",
            driver_number,
            lap_num,
            len(car_df),
        )

        if car_df.empty:
            logger.info(
                "[compute_lap_features_for_driver] driver=%s lap=%s: no car_data, SKIP",
                driver_number,
                lap_num,
            )
            continue

        feats = compute_features_for_lap(car_df)
        if feats is None:
            logger.info(
                "[compute_lap_features_for_driver] driver=%s lap=%s: feats=None, SKIP",
                driver_number,
                lap_num,
            )
            continue

        # 決定 phase
        phase = "middle"
        for ph, lap_list in phase_split.items():
            if lap_num in lap_list:
                phase = ph
                break

        row = {
            "driver": str(driver_number),
            "lap_number": int(lap_num),
            "phase": phase,
            "track_status": track_status,
            **feats,
        }
        out.append(row)

    if out:
        df_out = pd.DataFrame(out)
        logger.info(
            "[compute_lap_features_for_driver] driver=%s, feature laps=%s",
            driver_number,
            sorted(df_out["lap_number"].unique().tolist()),
        )
    else:
        logger.warning("[compute_lap_features_for_driver] driver=%s, NO laps with features", driver_number)

    return out


# ========== 降維 + 組成 LapFeature ==========

FEATURE_COLS = [
    "avg_throttle",
    "throttle_std",
    "full_throttle_ratio",
    "brake_ratio",
    "brake_intensity",
    "avg_gear",
    "gear_changes",
]


def embed_lap_features(
    lap_features: List[Dict[str, Any]],
    focus_laps: List[int],
) -> List[LapFeature]:
    """
    lap_features: 多位 driver 合併後的 list[dict]
    focus_laps: 想要特別標記的幾個 lap 編號（例如 [2, 27, 53]）
    """
    if not lap_features:
        return []

    df = pd.DataFrame(lap_features)

    X = df[FEATURE_COLS].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X_scaled)

    df["embed_x"] = X_embedded[:, 0]
    df["embed_y"] = X_embedded[:, 1]
    df["is_focus"] = df["lap_number"].isin(focus_laps)

    logger.info("[embed_lap_features] focus_laps = %s", focus_laps)

    results: List[LapFeature] = []
    for _, row in df.iterrows():
        lf = LapFeature(
            driver=row["driver"],
            lap_number=int(row["lap_number"]),
            embed_x=float(row["embed_x"]),
            embed_y=float(row["embed_y"]),
            avg_throttle=float(row["avg_throttle"]),
            throttle_std=float(row["throttle_std"]),
            full_throttle_ratio=float(row["full_throttle_ratio"]),
            brake_ratio=float(row["brake_ratio"]),
            brake_intensity=float(row["brake_intensity"]),
            avg_gear=float(row["avg_gear"]),
            gear_changes=int(row["gear_changes"]),
            phase=row["phase"],
            is_focus=bool(row["is_focus"]),
            track_status=row["track_status"],
        )
        results.append(lf)

    return results


# ========== 比賽 / 車手選擇相關 ==========

def parse_race_id(race_id: str) -> Tuple[int, str]:
    """
    把字串例如 '2024japan' 拆成 (year=2024, slug='japan')
    """
    race_id = race_id.strip()
    if len(race_id) < 5 or not race_id[:4].isdigit():
        raise ValueError(f"Invalid race_id format: {race_id!r}")
    year = int(race_id[:4])
    slug = race_id[4:]
    return year, slug


def expand_year_all(year: int) -> List[str]:
    """
    將 '2024all' 之類展開成該年份所有「正賽」的 race_id 列表。
    不再寫死 2024 清單，而是直接查 openF1 /sessions。

    回傳格式：
        ['2024bahrain', '2024saudiarabian', '2024australian', ...]
    slugs 主要由 meeting_name / location 經 _slugify_race 而來。
    """
    sessions_df = get_openf1("sessions", {"year": year, "session_type": "Race"})
    if sessions_df.empty:
        logger.warning("[expand_year_all] no Race sessions for year=%s", year)
        return []

    # 依日期排序，盡量接近實際年曆順序
    if "date_start" in sessions_df.columns:
        sessions_df["date_start"] = pd.to_datetime(
            sessions_df["date_start"], format="ISO8601", utc=True, errors="coerce"
        )
        sessions_df = sessions_df.sort_values("date_start")

    slugs: List[str] = []
    seen: set[str] = set()

    for _, row in sessions_df.iterrows():
        # 優先用 meeting_name，其次 location，再其次 country_name
        base = ""
        if "meeting_name" in sessions_df.columns and pd.notna(row.get("meeting_name", None)):
            base = str(row["meeting_name"])
        elif "location" in sessions_df.columns and pd.notna(row.get("location", None)):
            base = str(row["location"])
        elif "country_name" in sessions_df.columns and pd.notna(row.get("country_name", None)):
            base = str(row["country_name"])

        slug = _slugify_race(base)
        if not slug:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)

    logger.info("[expand_year_all] year=%s -> slugs=%s", year, slugs)
    return [f"{year}{slug}" for slug in slugs]


def parse_race_ids_from_file(path: str) -> List[str]:
    """
    從 driver_style_races.txt 讀取要處理的 race_id 列表。
    支援：
      - 逗號分隔，例如：2024japan, 2024china
      - 多行
      - # 開頭為註解
      - '2024all' 代表 2024 整季所有站（透過 openF1 動態展開）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"race list file not found: {path}")

    race_ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 去掉行尾註解
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in line.split(","):
                token = token.strip()
                if not token:
                    continue
                race_ids.append(token)

    expanded: List[str] = []
    seen = set()
    for rid in race_ids:
        m = re.match(r"^(\d{4})all$", rid.strip(), flags=re.IGNORECASE)
        if m:
            year = int(m.group(1))
            for rid2 in expand_year_all(year):
                if rid2 not in seen:
                    expanded.append(rid2)
                    seen.add(rid2)
        else:
            rid_norm = rid.strip()
            if rid_norm and rid_norm not in seen:
                expanded.append(rid_norm)
                seen.add(rid_norm)

    return expanded


def find_race_session(year: int, slug: str) -> Tuple[int, str]:
    """
    給 year + slug（例如 year=2024, slug='japan'），回傳：
      - 該年該分站「正賽」的 session_key
      - meeting_name（例如 'Japanese Grand Prix'）

    做法：
      1) 先打 /v1/meetings?year=year，用 meeting_name / location 做模糊比對，找出對應 meeting
      2) 用 meeting_key 去 /v1/sessions?meeting_key=... 找出 'Race' 那個 session
    """
    slug_norm = _slugify_race(slug)

    # 1) 用 meetings 找到那一場比賽 (meeting)
    meetings_df = get_openf1("meetings", {"year": year})
    if meetings_df.empty:
        raise ValueError(f"[find_race_session] no meetings found for year={year}")

    if "meeting_key" not in meetings_df.columns:
        raise ValueError(
            f"[find_race_session] meetings endpoint has no meeting_key column; columns={list(meetings_df.columns)}"
        )

    # 確認有 meeting_name，沒有的話就用其他欄位 fallback
    if "meeting_name" not in meetings_df.columns:
        name_col = None
    else:
        name_col = "meeting_name"

    # 建一個 slug 欄位來比對
    slug_candidates = []

    for idx, row in meetings_df.iterrows():
        # 先用 meeting_name
        base = ""
        if name_col is not None:
            base = _slugify_race(row[name_col])

        # 補上 location / country 之類的資訊當備援
        loc_parts = []
        for col in ["location", "country_name", "country_code"]:
            if col in meetings_df.columns and pd.notna(row.get(col, None)):
                loc_parts.append(str(row[col]))
        loc_slug = _slugify_race("".join(loc_parts))

        combined_slug = base + loc_slug
        slug_candidates.append(combined_slug)

    meetings_df["__slug"] = slug_candidates

    mask = meetings_df["__slug"].str.contains(slug_norm, na=False)
    candidates = meetings_df[mask]

    if candidates.empty:
        raise ValueError(
            f"[find_race_session] cannot match slug='{slug}' for year={year}; "
            f"available slugs={meetings_df[['meeting_key','__slug']].to_dict('records')}"
        )

    # 若有多場（例如同年兩場中國 GP），就用 date_start 排序取最後一場
    if "date_start" in candidates.columns:
        candidates["date_start"] = pd.to_datetime(
            candidates["date_start"],
            format="ISO8601",
            utc=True,
            errors="coerce",
        )
        candidates = candidates.sort_values("date_start")
    meeting_row = candidates.iloc[-1]

    meeting_key = int(meeting_row["meeting_key"])
    meeting_name = str(meeting_row.get("meeting_name", f"{year}-{slug}"))

    # 2) 用 meeting_key 找該場賽事的「正賽 session」
    sessions_df = get_openf1("sessions", {"meeting_key": meeting_key})
    if sessions_df.empty:
        raise ValueError(
            f"[find_race_session] no sessions found for meeting_key={meeting_key} (year={year}, slug={slug})"
        )

    race_df = sessions_df.copy()
    # 優先用 session_name == 'Race'
    if "session_name" in sessions_df.columns:
        name_u = sessions_df["session_name"].astype(str).str.upper()
        mask_race = name_u.eq("RACE")
        if mask_race.any():
            race_df = sessions_df[mask_race]

    # 如果還是有多筆，用 date_start 排最後一筆
    if "date_start" in race_df.columns:
        race_df["date_start"] = pd.to_datetime(
            race_df["date_start"],
            format="ISO8601",
            utc=True,
            errors="coerce",
        )
        race_df = race_df.sort_values("date_start")

    session_row = race_df.iloc[-1]
    session_key = int(session_row["session_key"])

    logger.info(
        "[find_race_session] year=%s slug=%s -> session_key=%s, meeting_name=%s",
        year,
        slug,
        session_key,
        meeting_name,
    )
    return session_key, meeting_name


def map_team_color(team_name: str) -> str:
    """
    根據隊名給一個代表色。若無法辨識則回傳灰色。
    """
    name = (team_name or "").lower()
    # 2023–2024 常見車隊（盡量 cover）
    if "red bull" in name:
        return "#1E5BC6"
    if "mercedes" in name:
        return "#00D2BE"
    if "ferrari" in name:
        return "#DC0000"
    if "mclaren" in name:
        return "#FF8700"
    if "aston martin" in name:
        return "#006F62"
    if "alpine" in name:
        return "#0090FF"
    if "williams" in name:
        return "#005AFF"
    if "alphatauri" in name or "rb " in name or "rb honda" in name:
        return "#2B4562"
    if "sauber" in name or "kick" in name or "stake" in name:
        return "#00E701"
    if "haas" in name:
        return "#B6BABD"
    # default
    return "#888888"


def select_race_drivers(session_key: int) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    從 /session_result + /drivers 推出：
      - 冠軍：所有「有名次的完賽者」中 position 最小者
      - 冠軍隊友：從 /drivers 找出與冠軍同隊、且 driver_number 不同的車手
                   （即使該隊友 DNF 也一樣納入比較）
      - 最後完賽車手：所有「有名次的完賽者」中 position 最大者

    回傳：
      - driver_numbers: [winner, teammate, last_finisher]（去重後仍維持這個優先順序）
      - drivers_meta: 每個 driver 的 {id,label,color,team_name}（一律從 /drivers 取隊名再 map 顏色）
    """
    # 1) 讀 session_result，專責名次 / 狀態
    res_df = get_openf1("session_result", {"session_key": session_key})
    if res_df.empty:
        raise ValueError(f"No session_result found for session_key={session_key}")

    if "driver_number" not in res_df.columns:
        raise ValueError(
            f"session_result has no driver_number column; columns={list(res_df.columns)}"
        )
    res_df["driver_number"] = res_df["driver_number"].astype(int)

    def pick_col(candidates: List[str], df: pd.DataFrame) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # position 欄位
    pos_col = pick_col(
        ["position", "position_order", "result_position", "classification_position"],
        res_df,
    )
    if pos_col is None:
        raise ValueError(
            f"session_result has no usable position column; columns={list(res_df.columns)}"
        )
    res_df[pos_col] = pd.to_numeric(res_df[pos_col], errors="coerce")

    # 狀態 / 圈數欄位（用來過濾「完賽者」）
    status_col = pick_col(["result_status", "status", "classification", "final_status"], res_df)

    # 2) 建立「完賽者 classified」：有 position，且不是 DNF / DSQ 等
    df = res_df.copy()
    if status_col is not None:
        status_u = df[status_col].astype(str).str.upper()
        bad_mask = status_u.str.contains(
            "DNF|DNS|DNQ|DSQ|DQ|RET|EXCLUDED|WITHDRAWN",
            regex=True,
        )
        classified = df[~bad_mask & df[pos_col].notna()].copy()
    else:
        classified = df[df[pos_col].notna()].copy()

    if classified.empty:
        raise ValueError(f"No classified results found for session_key={session_key}")

    # position 愈小名次愈前面
    classified = classified.sort_values(pos_col)

    # 3) 冠軍（一定是完賽者之一）
    winner_row = classified.iloc[0]
    winner_num = int(winner_row["driver_number"])

    # 4) 讀 /drivers，專責「隊伍與姓名」與找隊友
    drivers_df = get_openf1("drivers", {"session_key": session_key})
    if drivers_df.empty:
        raise ValueError(f"No drivers found for session_key={session_key}")

    if "driver_number" not in drivers_df.columns:
        raise ValueError(
            f"drivers endpoint has no driver_number column; columns={list(drivers_df.columns)}"
        )
    drivers_df["driver_number"] = drivers_df["driver_number"].astype(int)

    # 決定姓名欄與隊伍欄
    if "full_name" in drivers_df.columns:
        name_col_drv = "full_name"
    else:
        name_col_drv = None
        for c in ["broadcast_name", "driver_name", "name"]:
            if c in drivers_df.columns:
                name_col_drv = c
                break

    if "team_name" in drivers_df.columns:
        team_col_drv = "team_name"
    else:
        team_col_drv = None
        for c in ["constructor_name", "team"]:
            if c in drivers_df.columns:
                team_col_drv = c
                break

    if team_col_drv is None:
        # 沒有隊名欄位就補一欄 Unknown，避免後續崩潰
        team_col_drv = "team_name"
        drivers_df[team_col_drv] = "Unknown"

    # 冠軍的隊伍名稱（從 drivers 取）
    winner_drv_row = drivers_df.loc[drivers_df["driver_number"] == winner_num]
    if winner_drv_row.empty:
        winner_team_name = "Unknown"
    else:
        winner_team_name = str(winner_drv_row.iloc[0][team_col_drv])

    # 5) 冠軍隊友：同隊的另一位車手（即使 DNF 也照樣視為隊友）
    teammate_df = drivers_df[
        (drivers_df["driver_number"] != winner_num)
        & (drivers_df[team_col_drv] == winner_team_name)
    ].copy()

    if not teammate_df.empty:
        # 如果有兩位以上（理論上不會），就用 session_result 中名次較前者
        # 為了排序，需要把 session_result 的 position merge 進來
        teammate_df = teammate_df.merge(
            res_df[["driver_number", pos_col]],
            on="driver_number",
            how="left",
        )
        teammate_df[pos_col] = pd.to_numeric(teammate_df[pos_col], errors="coerce")
        teammate_df = teammate_df.sort_values(pos_col)
        teammate_row = teammate_df.iloc[0]
        teammate_num = int(teammate_row["driver_number"])
    else:
        # 找不到同隊隊友（資料缺 / 單車隊），退而求其次拿「第二名」當比較對象
        if len(classified) > 1:
            teammate_row = classified.iloc[1]
            teammate_num = int(teammate_row["driver_number"])
        else:
            teammate_num = winner_num  # 真的只剩一台車的極端情況

    # 6) 最後完賽車手：所有「完賽者」中 position 最大者
    last_row = classified.sort_values(pos_col, ascending=False).iloc[0]
    last_finisher_num = int(last_row["driver_number"])

    # 7) 組出 driver list，保持 [冠軍, 冠軍隊友, 最後完賽者] 的優先順序，但去掉重複
    ordered_nums = [winner_num, teammate_num, last_finisher_num]
    driver_numbers: List[int] = []
    seen_nums = set()
    for n in ordered_nums:
        if n not in seen_nums:
            driver_numbers.append(n)
            seen_nums.add(n)

    # 8) 組 drivers_meta（完全依賴 /drivers 的姓名與隊名，再 map 顏色）
    drivers_meta: List[Dict[str, Any]] = []
    for num in driver_numbers:
        row = drivers_df.loc[drivers_df["driver_number"] == num]
        if row.empty:
            label = f"Driver {num}"
            team_name = "Unknown"
        else:
            row = row.iloc[0]
            if name_col_drv is not None and name_col_drv in row.index:
                label = str(row[name_col_drv])
            else:
                label = f"Driver {num}"
            team_name = str(row[team_col_drv]) if team_col_drv in row.index else "Unknown"

        color = map_team_color(team_name)
        drivers_meta.append(
            {
                "id": str(num),
                "label": label,
                "team_name": team_name,
                "color": color,
            }
        )

    logger.info(
        "[select_race_drivers] session_key=%s winner=%s teammate=%s last_finisher=%s",
        session_key,
        winner_num,
        teammate_num,
        last_finisher_num,
    )
    return driver_numbers, drivers_meta


# ========== JSON builder ==========

def build_driver_style_embedding_json(
    session_key: int,
    driver_numbers: List[int],
    drivers_meta: List[Dict[str, Any]],
    focus_laps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    session_key：例如 2024 日本站 正賽的 session_key
    driver_numbers：本場要比較的 driver_number list（例如 [1,11,2]）
    drivers_meta：對應的 metadata（顏色、名稱）
    focus_laps：預設會依三等分自動選開局、中段、尾段各一圈
    """
    logger.info(
        "[build_driver_style_embedding_json] start session_key=%s, drivers=%s",
        session_key,
        driver_numbers,
    )

    # 先抓一次 race_control，後面 laps / status 都會用到
    rc_df = fetch_race_control(session_key)
    race_start = fetch_race_start_time(session_key, rc_df)

    # 用第一位車手的 laps 來決定全場圈號範圍與 phase split
    ref_driver = driver_numbers[0]
    laps_ref = fetch_laps(session_key, ref_driver, rc_df, race_start)
    min_lap = int(laps_ref["lap_number"].min())
    max_lap = int(laps_ref["lap_number"].max())

    logger.info(
        "[build_driver_style_embedding_json] ref_driver=%s, min_lap=%s, max_lap=%s",
        ref_driver,
        min_lap,
        max_lap,
    )

    # ===== 依 lap_number 三等分 =====
    total_laps = max_lap - min_lap + 1
    base_chunk = total_laps // 3
    remainder = total_laps % 3

    # 把餘數優先分配給前兩段，避免尾段太短
    opening_size = base_chunk + (1 if remainder > 0 else 0)
    middle_size = base_chunk + (1 if remainder > 1 else 0)
    ending_size = total_laps - opening_size - middle_size

    opening_start = min_lap
    opening_end = opening_start + opening_size - 1
    middle_start = opening_end + 1
    middle_end = middle_start + middle_size - 1
    ending_start = middle_end + 1
    ending_end = max_lap   # sanity: ending_start ~ max_lap

    opening_laps = list(range(opening_start, opening_end + 1))
    middle_laps = list(range(middle_start, middle_end + 1))
    ending_laps = list(range(ending_start, ending_end + 1))

    logger.info("[build_driver_style_embedding_json] phase_split plan =")
    logger.info("  opening: %s", opening_laps)
    logger.info("  middle : %s", middle_laps)
    logger.info("  ending : %s", ending_laps)

    phase_split = {
        "opening": opening_laps,
        "middle": middle_laps,
        "ending": ending_laps,
    }

    # 真正算特徵
    all_features: List[Dict[str, Any]] = []
    for drv in driver_numbers:
        logger.info("[build_driver_style_embedding_json] compute features for driver=%s", drv)
        feats = compute_lap_features_for_driver(session_key, drv, rc_df, phase_split)
        all_features.extend(feats)

    if not all_features:
        raise ValueError("No lap features computed. Check session_key / drivers / data availability.")

    df_all = pd.DataFrame(all_features)
    logger.info(
        "[build_driver_style_embedding_json] all feature laps (combined)=%s",
        sorted(df_all["lap_number"].unique().tolist()),
    )

    # ===== focus_laps：依「三等分後的 phase」各選一圈 =====
    if focus_laps is None:
        def pick_mid(laps: List[int]) -> Optional[int]:
            if not laps:
                return None
            return laps[len(laps) // 2]

        early_focus = pick_mid(opening_laps) or min_lap
        middle_focus = pick_mid(middle_laps) or ((min_lap + max_lap) // 2)
        ending_focus = ending_laps[-1] if ending_laps else max_lap

        focus_laps = [early_focus, middle_focus, ending_focus]

    lap_embeds = embed_lap_features(all_features, focus_laps)

    laps_json = [asdict(lf) for lf in lap_embeds]

    return {
        "drivers": drivers_meta,
        "laps": laps_json,
        "focus_laps": focus_laps,
        "feature_columns": FEATURE_COLS,
        "session_key": session_key,
        "min_lap": min_lap,
        "max_lap": max_lap,
    }


# ========== Logging & main ==========

def setup_logging_for_race(race_id: str) -> None:
    """
    設定 logging：
      - log/driver_style_{race_id}.log（DEBUG）
      - console（INFO）
    """
    os.makedirs("log", exist_ok=True)
    log_filename = os.path.join(
        "log",
        f"driver_style_{race_id}.log",
    )

    logger.setLevel(logging.DEBUG)

    # 檔案 handler
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    f_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(f_fmt)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    c_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(c_fmt)

    # 避免重複加 handler
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger.handlers.clear()
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info("===== Driver Style Map logging started for race_id=%s =====", race_id)
    logger.info("log file: %s", log_filename)


def process_single_race(year: int, slug: str, out_dir: str = ".") -> None:
    race_id = f"{year}{slug}"
    setup_logging_for_race(race_id)
    logger.info("=== Start processing %s ===", race_id)

    session_key, meeting_name = find_race_session(year, slug)
    logger.info(
        "[main] race_id=%s -> session_key=%s, meeting_name=%s",
        race_id,
        session_key,
        meeting_name,
    )

    driver_numbers, drivers_meta = select_race_drivers(session_key)

    result = build_driver_style_embedding_json(
        session_key=session_key,
        driver_numbers=driver_numbers,
        drivers_meta=drivers_meta,
    )

    os.makedirs(out_dir, exist_ok=True)
    output_filename = os.path.join(out_dir, f"driver_style_embedding_{race_id}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Saved %s", output_filename)
    print(f"[INFO] Saved {output_filename} (race_id={race_id})")


def main() -> None:
    config_path = "driver_style_races.txt"
    print(f"[INFO] Reading race list from {config_path} ...")
    race_ids = parse_race_ids_from_file(config_path)
    if not race_ids:
        print("[WARN] No race_id found in driver_style_races.txt")
        return

    print(f"[INFO] Races to process: {', '.join(race_ids)}")
    for race_id in race_ids:
        year, slug = parse_race_id(race_id)
        print(f"[INFO] Processing {race_id} ...")
        try:
            process_single_race(year, slug, out_dir=".")
        except Exception as e:
            print(f"[ERROR] Failed to process {race_id}: {e}")


if __name__ == "__main__":
    main()