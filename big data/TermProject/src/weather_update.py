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
    r = requests.get(BASE_URL, headers=HEADERS, params=params)

    if r.status_code != 200:
        print(f"[WARN] {date_str} 無法取得資料，HTTP {r.status_code}")
        return []

    return r.json().get("results", [])


# -------------------------------------------
# 處理資料
# -------------------------------------------
def process_daily_records(records):
    if not records:
        return None

    df = pd.DataFrame(records)
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

    return df


# -------------------------------------------
# 主更新函式
# -------------------------------------------
def update_weather_once():
    if not os.path.exists(OUTPUT):
        print("[INFO] 沒有資料檔案，建立一個新的。")
        df = pd.DataFrame()
        df.to_csv(OUTPUT)

    df = pd.read_csv(OUTPUT)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # 找到最後一天
    if len(df) == 0:
        last_date = datetime(2000, 1, 1).date()
    else:
        last_date = df["date"].max()

    next_date = last_date + timedelta(days=1)
    today = datetime.today().date()

    if next_date >= today:
        print("[INFO] 今日資料尚未更新，先睡覺。")
        return

    print(f"[INFO] 嘗試抓取 {next_date} 的資料...")

    records = fetch_one_day(str(next_date))
    processed = process_daily_records(records)

    if processed is None or len(processed) == 0:
        print(f"[INFO] NOAA 尚未更新 {next_date} 的資料。")
        return

    # append
    processed.to_csv(OUTPUT, mode="a", header=False)
    print(f"[SUCCESS] 已更新 {next_date}！")


# -------------------------------------------
# 無限迴圈：每天檢查一次
# -------------------------------------------
if __name__ == "__main__":
    print("[INFO] 啟動自動氣象更新器...")

    while True:
        update_weather_once()
        print("[INFO] 休息 12 小時...")
        time.sleep(60 * 60 * 12)
