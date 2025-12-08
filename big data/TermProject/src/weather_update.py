import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import date
import logging
import time

# -------------------------
# 設定
# -------------------------
load_dotenv()
TOKEN = os.getenv("NOAA_API_TOKEN")

STATION = "GHCND:USW00094728"  # New York Central Park
START_YEAR = 2020
END_YEAR = 2025  # 建議至少抓 5 年資料以達到 1000 筆

BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
HEADERS = {"token": TOKEN}

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


# -------------------------
# 抓單一 datatype（含 offset）
# -------------------------
def fetch_datatype(year, datatype):
    all_records = []

    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"

        if year == END_YEAR and month > date.today().month:
            break

        if month == 12:
            end = f"{year}-12-31"
        else:
            end = f"{year}-{month+1:02d}-01"

        offset = 1

        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": STATION,
                "startdate": start,
                "enddate": end,
                "datatypeid": datatype,
                "limit": 1000,
                "offset": offset,
            }

            try:
                r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)

                if r.status_code != 200:
                    log.warning(f"{datatype} {year}-{month:02d} 回應 {r.status_code}")
                    break

                results = r.json().get("results", [])

                # 沒資料就終止該月抓取
                if len(results) == 0:
                    break

                all_records.extend(results)
                offset += 1000  # 翻下一頁

            except Exception as e:
                log.warning(f"{datatype} {year}-{month:02d} 錯誤: {e}")
                time.sleep(1)
                break

        log.info(f"{year}-{month:02d} 抓到 {len(all_records)} 筆 {datatype}（累積）")

    return all_records


# -------------------------
# Main
# -------------------------
def main():
    log.info("開始抓取 NOAA 資料...")

    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        log.info(f"處理 {year} 年資料...")

        for dt in DATATYPES:
            log.info(f"抓取 {dt} ...")
            records = fetch_datatype(year, dt)
            all_data.extend(records)

    # -------------------------
    # 轉 DataFrame
    # -------------------------
    df = pd.DataFrame(all_data)

    if df.empty:
        log.error("抓不到任何資料，請檢查 API token 或網路。")
        return

    df["date"] = pd.to_datetime(df["date"]).dt.date

    df_pivot = df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="mean"
    )

    # -------------------------
    # 數據轉換：除以 10（氣溫/降雨量）
    # -------------------------
    for col in ["TMAX", "TMIN"]:
        if col in df_pivot.columns:
            df_pivot[col] = df_pivot[col] / 10.0

    if "PRCP" in df_pivot.columns:
        df_pivot["PRCP"] = df_pivot["PRCP"] / 10.0

    # -------------------------
    # 天氣旗標 WTxx（缺資料補 0）
    # -------------------------
    for wt in ["WT01", "WT03"]:
        if wt in df_pivot.columns:
            df_pivot[wt] = df_pivot[wt].fillna(0).apply(lambda x: 1 if x > 0 else 0)

    # -------------------------
    # 排序 & 輸出
    # -------------------------
    df_pivot = df_pivot.sort_index()
    df_pivot.to_csv("weather_processed.csv")

    log.info("處理完成！輸出檔案：weather_processed.csv")
    log.info(f"共 {len(df_pivot)} 天資料")
    print(df_pivot.head())


if __name__ == "__main__":
    main()
