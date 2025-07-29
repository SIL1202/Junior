### A. Research Topic:

**Detecting AI-Generated Speech Using Deep Learning and Spectral Analysis**
（使用深度學習與頻譜分析技術來辨識 AI 合成語音）

------

### B. Research Motivation:

隨著語音生成技術（如 Text-to-Speech, Voice Cloning）迅速進步，許多 AI 語音的真實度已能幾可亂真，可能被濫用於詐騙、偽造聲明等社會問題。然而，目前仍缺乏一套高準確率、低成本、可部署於終端的語音真偽鑑別技術。因此，我們希望透過深度學習結合聲音特徵分析，訓練一個模型來辨識語音是否為 AI 合成，以提升資訊安全防禦能力，也幫助社會大眾理解 AI 技術的風險與防範。

------

### C. Related Work:

1. **ASVspoof Challenge (2015, 2019, 2021)** – 國際語音安全競賽，提供語音欺騙資料集
   https://www.asvspoof.org/
2. **WaveFake: A Data Set to Facilitate Audio Deepfake Detection**
   https://arxiv.org/abs/2110.06666
3. **A Survey on Audio Deepfake Detection** (IEEE Access, 2021)
   https://ieeexplore.ieee.org/document/9440768
4. **Detecting Speech Deepfakes with RawNet2**
   https://arxiv.org/abs/2104.04040
5. **Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof)**
   提供不同語音攻擊類型（text-to-speech, voice conversion）

------

### D. Problems and Solutions:

**問題：**
隨著 AI 語音合成技術（如 Tacotron2、WaveNet、VALL-E）越來越強，普通人難以分辨真實語音與合成語音，現有的辨識方法仍面臨準確率不足、資料不足或實作門檻過高的問題。

**我們的解法：**

1. 利用公開資料集（如 ASVspoof、WaveFake）取得大量真實與合成語音樣本
2. 使用 `librosa` 等工具擷取語音特徵（如 MFCC、Mel-spectrogram）
3. 訓練 CNN 模型進行二分類任務（Real / AI Generated）
4. 分析模型準確率，並進一步觀察不同模型架構、資料增強方法是否能提升辨識成功率
5. 若有餘力，加入 adversarial testing 或多任務分類（如辨別是哪種生成器）

------

### F. Expected Results:

- 訓練出一個具有高準確率的語音辨識模型（>90% accuracy on test set）
- 可以將任意語音輸入模型，回傳是否為 AI 合成語音
- 最終 demo 預期能實現「輸入語音 → 顯示真假預測 + 視覺化頻譜圖」
- 若資料與模型設計良好，可拓展為 Web demo 或行動裝置部署版本

------

### G. Time Management and Division of Labor:

| 時間區間   | 任務內容                                                     |
| ---------- | ------------------------------------------------------------ |
| Week 1–4   | 相關文獻與資料集調查、工具環境建置（Python、PyTorch、librosa） |
| Week 5-8   | 初步模型訓練與資料前處理（頻譜擷取、圖像化）                 |
| Week 9-12  | 模型優化與評估（調整架構、增加資料、改進 loss function）     |
| Week 13-16 | 實驗分析與報告撰寫，製作成果展示 demo（可選：Web UI）        |
