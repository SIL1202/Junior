## 基礎機器學習關鍵專有名詞整理

AI
│
├── 機器學習 (ML)
│   ├── 傳統 ML（決策樹、KNN、SVM...）
│   └── 深度學習 (DL)
│       ├── CNN：影像處理、分類、辨識
│       ├── RNN / LSTM：語音、時間序列
│       └── Transformer：語言模型、圖像生成
│
└── 應用層（應用 AI 到各種問題）
    ├── 圖像處理（Computer Vision）
    ├── 自然語言處理（NLP）
    ├── 強化學習（遊戲、機器人）
    └── 多模態（文字+圖片、影片字幕等）

### 1. Model (模型)

機器學習中的模型指的是一組數學函數，用來根據輸入資料預測輸出。舉例：線性回歸模型、神經網路模型。

### 2. Hypothesis Set (假設集合, $\mathcal{H}$)

所有可能模型的集合。假設集合越大，模型選擇越多，但越容易 overfitting。

### 3. Overfitting（過擬合）

模型在訓練資料表現很好，但在新資料上表現變差的現象。原因通常是模型太複雜、資料量太少。

### 4. Loss Function（損失函數）

衡量模型預測與真實答案差距的函數。常見如：MSE（均方誤差）、Cross-Entropy（交叉熵）。

### 5. Arg min（argument of the minimum）

使某函數取得最小值的輸入值。寫法：

$$
\arg \min_{h \in \mathcal{H}} L(h, D)
$$

意思是「找出讓 Loss 最小的那個模型 h」。

### 6. Generalization（泛化）

模型學習後對未見過資料的預測能力。與 Overfitting 相對。

### 7. Training Data（訓練資料）

用來訓練模型的資料集。

### 8. Validation Data（驗證資料）

用來調整模型超參數與選擇模型的資料集。

### 9. Test Data（測試資料）

用來評估模型最終表現的資料集，不參與訓練與調整。

### 10. Learning Rate（學習率）

控制每次參數更新的步伐大小。學習率太大或太小都可能導致問題。

### 11. Batch Size（批次大小）

一次丟進模型訓練的樣本數。影響訓練速度與效果。

### 12. Epoch

訓練資料被完整跑過一次的過程。

### 13. Optimizer（優化器）

更新模型參數的方法。常見如 SGD、Adam。

### 14. Activation Function（激活函數）

增加模型非線性能力的函數。常見如 ReLU、Sigmoid、Tanh。

### 15. GPU（圖形處理單元）

用來加速深度學習運算的硬體，特別適合平行計算。

