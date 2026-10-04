# 題目二：Informer 與 Transformer 長序列預測

本資料夾包含題目二的完整程式碼。主入口為 `Informer_Transformer.ipynb`；若偏好命令列，也可使用 `run_experiments.py`。

## 實驗設計

- Dataset：Household Electric Power Consumption（Kaggle slug：`uciml/electric-power-consumption-data-set`）
- 預測目標：`Global_active_power`
- 特徵：資料集的 7 個數值欄位，加上 6 個週期時間特徵
- 缺失值：15 分鐘重採樣後，以時間內插，再以前／後值補齊邊界
- 切分：依時間順序 70% training、15% validation、15% test
- 標準化：只使用 training 部分計算 mean/std
- Input Length：96、336、672（約 1 天、3.5 天、1 週）
- Prediction Horizon：96（未來 1 天）
- Sliding-window stride：4（每小時建立一個樣本）

主比較對每個 Input Length 都訓練：

1. 自行實作的 Vanilla Transformer：value/time/position embedding、Multi-head Attention、Encoder、Decoder、Prediction Head。
2. 依官方 Informer2020 架構改寫成現代 PyTorch 版本的 Informer。

在最長的 Input Length 672 上另外進行三項 Informer 消融：

1. ProbSparse Self-Attention 改成 Full Attention。
2. 移除 Self-Attention Distilling。
3. 移除 Generative Decoder，改成 encoder-only direct multi-step prediction head。

## 執行方式

建議先在 Notebook 將 `QUICK_MODE=True` 跑通流程，再改成 `False` 正式訓練。正式實驗需要 CUDA 版 PyTorch；CPU 仍可執行，但長序列實驗會非常慢。

```powershell
python run_experiments.py --quick
python run_experiments.py
```

若已自行下載資料，請放在：

`data/household_power_consumption.txt`

否則程式會先嘗試透過 KaggleHub 下載公開 Kaggle dataset；失敗時再使用 UCI 官方資料來源。

## 輸出

正式執行後，所有產物都會保留在本資料夾內：

- `outputs/checkpoints/`：每組模型權重
- `outputs/predictions/`：相同 test set 的預測樣本
- `outputs/figures/`：loss、預測、長序列比較與消融圖
- `outputs/results.csv`：所有指定指標
- `outputs/histories.json`：每個 epoch 的 loss 與時間
- `outputs/experiment_manifest.json`：固定實驗設定

## Reference

- [Informer2020 官方程式庫](https://github.com/zhouhaoyi/Informer2020)
- [Kaggle Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
- [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

