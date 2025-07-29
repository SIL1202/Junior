import matplotlib.pyplot as plt
import linear_regression as training

radio = [
    37.8,
    39.3,
    45.9,
    41.3,
    10.8,
    9.0,
    48.9,
    32.5,
    20.0,
    5.5,
    29.7,
    1.5,
    21.0,
    24.0,
    17.2,
    31.5,
    11.0,
    36.9,
    14.2,
    23.1,
]

sales = [
    22.1,
    10.4,
    18.3,
    18.5,
    7.2,
    4.8,
    20.2,
    14.9,
    8.0,
    3.3,
    12.5,
    1.6,
    9.7,
    11.5,
    7.9,
    14.6,
    6.1,
    17.4,
    5.3,
    10.7,
]

initial_weight = 0.0
initial_bias = 0.0
learning_rate = 0.0005
iterations = 500

final_weight, final_bias, _ = training.training(
    radio=radio,
    sales=sales,
    weight=initial_weight,
    bias=initial_bias,
    iters=iterations,
    lr=learning_rate,
)

print("Final weight:", final_weight)
print("Final bias:", final_bias)

import matplotlib.pyplot as plt  # 匯入畫圖套件

# 1️⃣ 畫出訓練資料點（實際觀察到的資料）
plt.scatter(radio, sales, color="blue", label="Actual Data")
# → 使用 scatter plot（散點圖）
# → x 軸是 radio，y 軸是 sales
# → 用藍色點點顯示

# 2️⃣ 建立一條 x 軸上的連續數列（模擬從 0 到 60 的廣告預算）
x_line = list(range(0, 60))

# 3️⃣ 用你的模型公式計算對應的預測值
y_line = [final_weight * x + final_bias for x in x_line]
# → 這裡的公式就是：y = wx + b

# 4️⃣ 畫出預測線（紅色）
plt.plot(x_line, y_line, color="red", label="Prediction Line")
# → 這是線圖，把你剛剛算出來的預測點畫成一條直線

# 5️⃣ 設定 x/y 軸名稱、標題與圖例
plt.xlabel("Radio ($)")  # x 軸：Radio 廣告預算
plt.ylabel("Sales")  # y 軸：預測的銷售額
plt.title("Linear Regression Fit")  # 標題
plt.legend()  # 顯示 label（顯示藍點 vs 紅線）
plt.grid(True)  # 加上背景格線（好看）

# 6️⃣ 顯示圖形
plt.show()

predicted_sales = [final_weight * x + final_bias for x in radio]
residuals = [actual - predicted for actual, predicted in zip(sales, predicted_sales)]

plt.scatter(radio, residuals)
plt.xlabel("Radio ($)")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.axhline(y=0, color="red", lineStyles="--")
plt.show()
