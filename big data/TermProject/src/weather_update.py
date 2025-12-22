import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("NOAA_API_TOKEN")
STATION = "GHCND:USW00094728"
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
HEADERS = {"token": TOKEN}

OUTPUT = "../datasets/weather_processed.csv"


def fetch_one_day(date_str):
    params = {
        "datasetid": "GHCND",
        "stationid": STATION,
        "startdate": date_str,
        "enddate": date_str,
        "limit": 1000,
    }
    try:
        r = requests.get(BASE_URL, headers=HEADERS, params=params)
        if r.status_code != 200:
            print(f"[WARN] {date_str} 無法取得資料，HTTP {r.status_code}")
            return []
        return r.json().get("results", [])
    except Exception as e:
        print(f"[WARN] 連線錯誤: {e}")
        return []


def process_daily_records(records):
    if not records:
        return None

    df = pd.DataFrame(records)
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="mean"
    )

    # 溫度除以10
    if "TMAX" in df.columns:
        df["TMAX"] = df["TMAX"] / 10.0
    if "TMIN" in df.columns:
        df["TMIN"] = df["TMIN"] / 10.0
    if "PRCP" in df.columns:
        df["PRCP"] = df["PRCP"] / 10.0

    # Reset index to make 'date' a column again
    df = df.reset_index()
    return df


def update_weather_once():
    file_exists = os.path.exists(OUTPUT)
    existing_df = pd.DataFrame()
    last_date = datetime(2000, 1, 1).date()

    if file_exists:
        try:
            existing_df = pd.read_csv(OUTPUT)
            if "date" in existing_df.columns and not existing_df.empty:
                existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
                last_date = existing_df["date"].max()
            else:
                print("[WARN] 資料檔存在但無法讀取日期，重新開始。")
                file_exists = False
        except Exception as e:
            print(f"[WARN] 讀取資料檔失敗 ({e})，將重新建立。")
            file_exists = False

    next_date = last_date + timedelta(days=1)
    today = datetime.today().date()

    if next_date >= today:
        print("[INFO] 今日資料尚未更新，先睡覺。")
        return False

    print(f"[INFO] 嘗試抓取 {next_date} 的資料...")

    records = fetch_one_day(str(next_date))
    processed = process_daily_records(records)

    # 如果當天沒資料，建立一個只有日期的空資料，以免卡住
    if processed is None or processed.empty:
        print(f"[WARN] {next_date} 無資料，寫入空紀錄以跳過。")
        processed = pd.DataFrame({"date": [next_date]})

    # --- 修正重點：使用 concat 處理欄位不一致的問題 ---
    if not existing_df.empty:
        # 使用 pd.concat 合併新舊資料，Pandas 會自動對齊欄位，缺少的會補 NaN
        combined = pd.concat([existing_df, processed], ignore_index=True)
    else:
        combined = processed

    # 每次都完整寫入，確保欄位正確
    combined.to_csv(OUTPUT, index=False)

    print(f"[SUCCESS] 已更新 {next_date}！")
    return True


if __name__ == "__main__":
    print("[INFO] 啟動自動氣象更新器...")

    while True:
        try:
            updated = update_weather_once()
            if updated:
                # 如果有更新資料，只休息 0.5 秒就繼續抓下一天
                time.sleep(0.5)
            else:
                # 如果已經是最新的，休息 12 小時
                print("[INFO] 休息 12 小時...")
                time.sleep(60 * 60 * 12)
        except Exception as e:
            print(f"[ERROR] 發生未預期錯誤: {e}")
            print("休息 10 秒後重試...")
            time.sleep(10)
