# Weather Prediction Term Project

這是一個大數據分析期末專案，旨在使用歷史氣象資料預測未來一週的最高氣溫 (TMAX)。專案包含自動化資料爬蟲以及多種機器學習模型的比較分析。

## 專案架構

```
.
├── TermProject.pdf      # 專案需求說明
├── environment.yml      # Conda 環境設定檔
├── README.md            # 專案說明文件
├── datasets/            # 存放資料集與預測結果
│   └── weather_processed.csv  # 歷史氣象資料
└── src/                 # 程式碼
    ├── .env             # 環境變數 (需自行建立)
    ├── weather_update.py # NOAA 資料爬蟲與自動更新程式
    └── predict.py        # 機器學習訓練與預測主程式
```

---

## 環境安裝

本專案使用 Conda 進行環境管理。請依照以下步驟建立執行環境：

1. **建立環境**：
   
   ```bash
   conda env create -f environment.yml
   ```
   
2. **啟用環境**：
   
   ```bash
   conda activate term
   ```

----

## 設定 (Configuration)

本專案使用 NOAA API 下載資料。在執行爬蟲之前，請在 `src/` 目錄下建立 `.env` 檔案，並填入您的 API Token：

**檔案路徑：** `src/.env`

```env
NOAA_API_TOKEN=你的_NOAA_API_TOKEN_貼在這裡
```

> 如果沒有 Token，請至 [NOAA NCDC](https://www.ncdc.noaa.gov/cdo-web/token) 申請。

---

## 使用說明

### 1. 更新氣象資料 (Data Collection)
若需取得最新資料，請執行更新程式。該程式會讀取現有 CSV 並從 NOAA 下載缺失日期的資料。

```bash
cd src
python weather_update.py
```
> 註：此程式預設為無窮迴圈（每天檢查一次），若只需更新一次，執行後待出現 `[INFO] 休息 12 小時...` 即可按 `Ctrl+C` 結束。

### 2. 執行預測模型 (Analysis & Forecasting)
執行主程式以訓練模型、評估成效並產出未來預測。

```bash
cd src
python predict.py
```

### 輸出結果
執行 `predict.py` 後，將在 `datasets/` 目錄下產生以下檔案：
*   `forecast_output.csv`: 測試集上的各模型預測結果比對。
*   `future_7days.csv`: 未來 7 天的氣溫預測。
*   **圖表視窗**: 包含實際 vs 預測走勢圖、誤差分佈圖、以及不同模型的 MAE 比較。

---

## 使用模型

本專案實作並比較了以下演算法：
*   **Baseline**: 移動平均 (MA), 指數平滑 (ES), 自回歸 (AR)
*   **Machine Learning**: 線性回歸 (Linear Regression), 隨機森林 (Random Forest)
