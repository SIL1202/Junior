import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ------------------------
# read data
# ------------------------
df = pd.read_csv("weather_processed.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


print(f"資料期間：{df['date'].min()} 到 {df['date'].max()}")
print(f"總筆數：{len(df)}")


# ------------------------
# 加入 lag 特徵
# ------------------------
def add_lag(df, columns, lags=[1, 2, 3, 7]):
    temp = df.copy()
    for col in columns:
        for lag in lags:
            temp[f"{col}_lag{lag}"] = temp[col].shift(lag)
    return temp


# 原始強特徵（全部包含）
strong_features = [
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

df = add_lag(df, strong_features, lags=[1, 2, 3])

# ------------------------
# 補季節特徵
# ------------------------
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

df = df.fillna(0)

# ------------------------
# 分割資料
# ------------------------
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

y_train = train["TMAX"].values
y_test = test["TMAX"].values


# ------------------------
# 組合全部特徵 X
# ------------------------
lag_features = [c for c in df.columns if "_lag" in c]
static_features = [
    "month",
    "day_of_year",
    "is_winter",
    "is_summer",
    "SNOW",
    "SNWD",
    "AWND",
    "WSF2",
    "WDF2",
    "WT01",
    "WT03",
]

feature_cols = lag_features + static_features

X_train = train[feature_cols].values
X_test = test[feature_cols].values

print(f"使用特徵數量：{len(static_features)} 個")


# ------------------------
# baseline 模型
# ------------------------
def moving_average(series, n_test, w=7):
    hist = list(series)
    preds = []

    for _ in range(n_test):
        pred = np.mean(hist[-w:])
        preds.append(pred)
        hist.append(pred)  # 更新視窗

    return np.array(preds)


def exponential_smoothing(series, n_test, alpha=0.3):
    last = series[-1]
    preds = []
    smoothed = last

    for _ in range(n_test):
        preds.append(smoothed)
        smoothed = alpha * smoothed + (1 - alpha) * preds[-1]
    return np.array(preds)


def autoregression(series, n_test, lag=3):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag : i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    hist = list(series[-lag:])
    preds = []
    for _ in range(n_test):
        pred = model.predict(np.array(hist[-lag:]).reshape(1, -1))[0]
        preds.append(pred)
        hist.append(pred)
    return np.array(preds)


ma_pred = moving_average(y_train, len(test))
es_pred = exponential_smoothing(y_train, len(test))
ar_pred = autoregression(y_train, len(test))


# ------------------------
# ML 模型
# ------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# ------------------------
# 評估
# ------------------------
def evaluate(name, true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    return mae


# 預測全部結果
def predict():
    pred_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual_TMAX": y_test,
            "pred_TMAX": ma_pred,
            "diff": y_test - ma_pred,
        }
    )
    print(pred_df.head(20))

    pred_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual_TMAX": y_test,
            "pred_TMAX": es_pred,
            "diff": y_test - es_pred,
        }
    )
    print(pred_df.head(20))

    pred_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual_TMAX": y_test,
            "pred_TMAX": ar_pred,
            "diff": y_test - ar_pred,
        }
    )
    print(pred_df.head(20))

    pred_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual_TMAX": y_test,
            "pred_TMAX": rf_pred,
            "diff": y_test - rf_pred,
        }
    )
    print(pred_df.head(20))

    pred_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual_TMAX": y_test,
            "pred_TMAX": lr_pred,
            "diff": y_test - lr_pred,
        }
    )
    print(pred_df.head(20))

    pred_df.to_csv("forecast_output.csv", index=False)


predict()
results = {
    "MA": evaluate("移動平均", y_test, ma_pred),
    "ES": evaluate("指數平滑", y_test, es_pred),
    "AR": evaluate("自回歸", y_test, ar_pred),
    "LR": evaluate("線性回歸", y_test, lr_pred),
    "RF": evaluate("隨機森林", y_test, rf_pred),
}

print("\n最佳模型：", min(results, key=results.get))


def forecast_next_week(df, model, feature_cols):
    future_preds = []
    temp_df = df.copy()

    for _ in range(7):
        last_row = temp_df.iloc[-1]
        next_date = last_row["date"] + pd.Timedelta(days=1)

        new_row = {"date": next_date}

        # 季節特徵
        new_row["month"] = next_date.month
        new_row["day_of_year"] = next_date.timetuple().tm_yday
        new_row["is_winter"] = int(next_date.month in [12, 1, 2])
        new_row["is_summer"] = int(next_date.month in [6, 7, 8])

        for col in ["SNOW", "SNWD", "AWND", "WSF2", "WDF2", "WT01", "WT03"]:
            if col in temp_df.columns:
                new_row[col] = temp_df[col].iloc[-1]
            else:
                new_row[col] = 0

        # lag 特徵
        for col in strong_features:
            for lag in [1, 2, 3]:
                new_row[f"{col}_lag{lag}"] = temp_df[col].iloc[-lag]

        next_df = pd.DataFrame([new_row]).fillna(0)

        # 預測
        X_next = next_df[feature_cols].values
        pred_tmax = model.predict(X_next)[0]
        future_preds.append((next_date, pred_tmax))

        next_df["TMAX"] = pred_tmax
        next_df["TMIN"] = temp_df["TMIN"].iloc[-1]
        next_df["PRCP"] = 0

        temp_df = pd.concat([temp_df, next_df], ignore_index=True)

    return pd.DataFrame(future_preds, columns=["date", "pred_TMAX"])


future_week = forecast_next_week(df, rf, feature_cols)
future_week.to_csv("future_7days.csv", index=False)

print("\n=== 未來七天溫度預測 ===")
print(future_week)

# ------------------------
# 視覺化
# ------------------------
plt.figure(figsize=(16, 10))

# 1. RF vs 實際
plt.subplot(2, 2, 1)
plt.plot(test["date"], y_test, label="實際 TMAX")
plt.plot(test["date"], rf_pred, label="RF 預測")
plt.xlabel("日期")
plt.ylabel("最高溫 (°F)")
plt.title("隨機森林預測結果")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# 2. 誤差分布
plt.subplot(2, 2, 2)
errors = rf_pred - y_test
plt.hist(errors, bins=30, edgecolor="black")
plt.xlabel("誤差 (°F)")
plt.ylabel("頻率")
plt.title("誤差分布 (RF)")
plt.grid(True)

# 3. MAE 比較
plt.subplot(2, 2, 3)
names = list(results.keys())
vals = list(results.values())
plt.bar(names, vals, color="skyblue")
plt.ylabel("MAE (°F)")
plt.title("模型 MAE 比較")
plt.grid(axis="y")

# 4. 殘差散佈圖
plt.subplot(2, 2, 4)
plt.scatter(rf_pred, errors, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("預測值 (°F)")
plt.ylabel("殘差")
plt.title("RF 殘差圖")
plt.grid(True)

plt.tight_layout()
plt.show()
lt.show()
