import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("NOAA_API_TOKEN")

STATION = "GHCND:USW00094728"  # New York Central Park
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
HEADERS = {"token": TOKEN}

# 你抓的所有 datatype
DATATYPES = [
    "TMAX",
    "TMIN",
    "PRCP",
    "SNOW",
    "SNWD",
    "AWND",
    "WSF2",
    "WDF2",
    "WT01",
    "WT03",
]

OUTPUT = "weather_processed.csv"


def fetch_range(start, end):
    """抓取 start ~ end 之間的所有天數資料"""
    params = {
        "datasetid": "GHCND",
        "stationid": STATION,
        "startdate": start,
        "enddate": end,
        "limit": 1000,
        "datatypeid": ",".join(DATATYPES),
    }
    r = requests.get(BASE_URL, headers=HEADERS, params=params)

    if r.status_code != 200:
        print("API ERROR:", r.text)
        return []

    return r.json().get("results", [])


def main():
    # 如果檔案不存在 → 直接退出
    if not os.path.exists(OUTPUT):
        print(f"[ERROR] 找不到 {OUTPUT}，請先跑一次 forecasting")
        return

    df = pd.read_csv(OUTPUT)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    last_date = df["date"].max()
    today = datetime.today().date()

    if last_date >= today:
        print("[INFO] 資料已是最新，不需更新")
        return

    print(f"[INFO] 目前資料到 {last_date}，開始更新到 {today}")

    start = last_date + timedelta(days=1)
    end = today

    raw = fetch_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if not raw:
        print("[INFO] 沒抓到新資料")
        return

    new_df = pd.DataFrame(raw)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

    pivot = new_df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="mean"
    )

    # 轉換單位
    for col in ["TMAX", "TMIN", "PRCP"]:
        if col in pivot.columns:
            pivot[col] /= 10.0

    pivot = pivot.reset_index()

    # 合併
    final = pd.concat([df, pivot], ignore_index=True)
    final = final.sort_values("date")

    final.to_csv(OUTPUT, index=False)
    print(f"[SUCCESS] 已更新 {len(pivot)} 天 → {OUTPUT}")


if __name__ == "__main__":
    main()
