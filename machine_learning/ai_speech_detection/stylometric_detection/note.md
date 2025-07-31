## 非神經網路（non-neural），使用（stylometric features）＋提升樹模型（LightGBM）來進行 AI 生成文本的（MGT, Machine-Generated Text）二元分類

### 1. 方法架構
 - 使用 stylometry：從文本中提取數千個語言學特徵（n-gram、詞性、依存關係、實體等），基於 spaCy。
 - 使用 LightGBM 分類器：因為它訓練快、解釋性強、不像神經網路黑盒。
 - 使用大量資料訓練：收集了 56 萬筆資料，來自許多資料集如 HC3、MAGE、Multitude、PAN’25。

### 2. 特徵設計（Feature Engineering）
 - 特徵來自：
    - lemma n-grams（詞元）
    - POS n-grams（詞性標記）
    - dependency bigrams（語法依存）
    - morph annotations（形態學標註）
 - 不使用預定義小型特徵集（如 StyloMetrix），而是大量語言特徵。

### 3. 實驗與結果
- 模型分為：small / medium / big（根據樹深、葉節點數等）
- 加入 Cross-validation 能顯著提升效果
- 在 TIRA 平台的測試表現（Test set）接近 baseline（TF-IDF），但未超越最強模型（神經網路為主）

### 4. 優點與貢獻
- 解釋性高：不像深度學習是黑盒，這種方法可以告訴你哪些語言特徵有用。
- 訓練快速：LGBM 不需要 GPU 就能訓練得很好。
- 不依賴 LLM：比起需要存取大型語言模型的內部資訊（logits、embedding）的白盒方法，這是純黑盒資料驅動做法。

### 5. Linker
 - [StylOch at PAN](./paper.pdf)

