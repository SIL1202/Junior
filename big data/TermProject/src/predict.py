import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")

# ------------------------
# 1. 讀取與前處理
# ------------------------
df = pd.read_csv("../datasets/weather_processed.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"資料期間：{df['date'].min()} 到 {df['date'].max()}")


# ------------------------
# 2. 特徵工程 (全標籤輔助版)
# ------------------------
def create_tags(data):
    """
    統一管理標籤定義
    """
    # 暖冬標籤 (>= 5度)
    data["is_warm"] = (data["TMAX"] >= 5.0).astype(int)
    # 冰凍標籤 (< 0度)
    data["is_freezing"] = (data["TMAX"] < 0.0).astype(int)

    # 降雨標籤
    if "PRCP" in data.columns:
        data["is_rainy"] = (data["PRCP"] > 0).astype(int)
    else:
        data["is_rainy"] = 0

    # 降雪標籤
    if "SNOW" in data.columns:
        data["is_snowy"] = (data["SNOW"] > 0).astype(int)
    else:
        data["is_snowy"] = 0

    return data


df = create_tags(df)

# 定義要產生 Lag 的特徵
features_to_lag = [
    "TMAX",
    "TMIN",
    "PRCP",
    "SNOW",
    "AWND",
    "is_warm",
    "is_freezing",
    "is_rainy",
    "is_snowy",
]


def add_lag(df, columns, lags=[1, 2, 3]):
    temp = df.copy()
    for col in columns:
        if col not in temp.columns:
            temp[col] = 0
        for lag in lags:
            temp[f"{col}_lag{lag}"] = temp[col].shift(lag)
    return temp


df = add_lag(df, features_to_lag, lags=[1, 2, 3])

# 季節特徵
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

df = df.fillna(0)

# ------------------------
# 3. 分割資料 (只用冬季)
# ------------------------
winter_months = [11, 12, 1, 2, 3]
winter_df = df[df["month"].isin(winter_months)].reset_index(drop=True)

print(f"冬季資料筆數：{len(winter_df)}")

lag_features = [c for c in df.columns if "_lag" in c]
static_features = ["month", "day_of_year", "is_winter"]
feature_cols = lag_features + static_features

X = winter_df[feature_cols].values
y = winter_df["TMAX"].values

split = int(len(winter_df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_dates = winter_df["date"].iloc[split:].reset_index(drop=True)

# ------------------------
# 4. 模型訓練
# ------------------------
print("\n[INFO] 正在訓練模型 (包含誤差準確率分析)...")
rf = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ------------------------
# 5. 評估 (加入新指標)
# ------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# [NEW] 計算誤差容許準確率 (Accuracy within range)
# 我們定義：如果預測值跟實際值相差在 3度C 以內，就算「準確」
tolerance = 5.0
within_tolerance = np.abs(y_pred - y_test) <= tolerance
tolerance_acc = np.mean(within_tolerance)

# [NEW] 計算衍生分類準確率 (Derived Classification Accuracy)
# 看看回歸模型預測出來的 "暖/冷" 狀態對不對
actual_warm = (y_test >= 5.0).astype(int)
pred_warm = (y_pred >= 5.0).astype(int)
class_acc = accuracy_score(actual_warm, pred_warm)

print(f"\n=== 最終模型評估 ===")
print(f"MAE (平均誤差)      : {mae:.2f} °C")
print(f"RMSE (均方根誤差)   : {rmse:.2f} °C")
print(f"R² (解釋力)         : {r2:.4f}")
print("-" * 30)
print(f"Tolerance Accuracy (±{tolerance}°C 準確率) : {tolerance_acc:.2%}")
print(f"Warm Day Recognition (暖日辨識率)       : {class_acc:.2%}")
print("-" * 30)

# 儲存
output_df = pd.DataFrame(
    {
        "date": test_dates,
        "actual_TMAX": y_test,
        "pred_TMAX": y_pred,
        "diff": y_test - y_pred,
        "within_3deg": within_tolerance,
    }
)
output_df.to_csv("../datasets/forecast_output.csv", index=False)


# ------------------------
# 6. 未來 7 天預測
# ------------------------
def forecast_next_week_final(df, model, feature_cols):
    future_preds = []
    temp_df = df.copy()
    temp_df = create_tags(temp_df)

    print(f"\n=== 未來 7 天預測 ===")

    for i in range(1, 8):
        last_row = temp_df.iloc[-1]
        next_date = last_row["date"] + pd.Timedelta(days=1)

        new_row = {"date": next_date}
        new_row["month"] = next_date.month
        new_row["day_of_year"] = next_date.dayofyear
        new_row["is_winter"] = int(next_date.month in [12, 1, 2])

        for col in features_to_lag:
            for lag in [1, 2, 3]:
                idx = -lag
                val = temp_df[col].iloc[idx] if abs(idx) <= len(temp_df) else 0
                new_row[f"{col}_lag{lag}"] = val

        next_df = pd.DataFrame([new_row]).fillna(0)
        for f in feature_cols:
            if f not in next_df.columns:
                next_df[f] = 0

        X_next = next_df[feature_cols].values
        pred_tmax = model.predict(X_next)[0]

        # 推導狀態
        is_warm = "Yes" if pred_tmax >= 5.0 else "No"
        desc = f"{pred_tmax:.2f}°C (暖日: {is_warm})"
        print(f"{next_date}: {desc}")

        future_preds.append(
            {"date": next_date, "pred_TMAX": pred_tmax, "is_warm": is_warm}
        )

        # 更新
        next_df["TMAX"] = pred_tmax
        next_df = create_tags(next_df)  # 自動補上所有標籤

        # 數值延續
        next_df["TMIN"] = temp_df["TMIN"].iloc[-1]
        next_df["PRCP"] = temp_df["PRCP"].iloc[-1]
        next_df["SNOW"] = temp_df["SNOW"].iloc[-1]
        next_df["AWND"] = temp_df["AWND"].iloc[-1]

        temp_df = pd.concat([temp_df, next_df], ignore_index=True)

    return pd.DataFrame(future_preds)


future_week = forecast_next_week_final(df, rf, feature_cols)
future_week.to_csv("future_7days.csv", index=False)

# ------------------------
# 7. 視覺化 (誤差準確率圖)
# ------------------------
plot_days = 150
last_dates = test_dates[-plot_days:]
last_y_test = y_test[-plot_days:]
last_y_pred = y_pred[-plot_days:]
last_within = within_tolerance[-plot_days:]

plt.figure(figsize=(14, 7))

# 畫實際溫度
plt.plot(last_dates, last_y_test, color="gray", alpha=0.5, label="Actual")
plt.plot(last_dates, last_y_pred, color="dodgerblue", label="Predicted")

# 標出「準確」的區間 (綠色背景)
# 這樣可以看出哪些天是準的，哪些天是不準的
for date, correct in zip(last_dates, last_within):
    if not correct:
        plt.axvline(x=date, color="red", alpha=0.1)  # 預測不準的日子畫紅底

plt.title(f"Prediction Accuracy (Red Zones = Error > {tolerance}°C)")
plt.legend()
plt.tight_layout()
plt.show()
