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

ProbSparse 的 query 選擇使用官方的 `max(sampled_scores) - sum(sampled_scores) / key_length`，取樣與 query 數量使用 `factor * ceil(log(length))`。Decoder self-attention 預設啟用官方 `mix=True` 的 transpose-before-reshape 行為；Encoder self-attention 與 Decoder cross-attention 不啟用 mix。

每組仍執行指定的 epochs，訓練結束後載入 validation MSE 最低的 checkpoint，再評估 test set。`results.csv` 會記錄 `best_epoch` 與 `best_validation_mse_standardized`，loss history 則保留全部 epochs。本次修改之前的 checkpoint 會自動判定為不相容並重新訓練。更新程式後，請重新啟動 Notebook kernel 並從頭執行；既有 Notebook 圖表與指標仍是舊版結果。

## 輸出

所有模型預設使用 `ReduceLROnPlateau`，每個 epoch 結束後以 validation 的 standardized MSE 調整 learning rate。初始值為 `1e-4`、`factor=0.5`、`patience=2`、相對改善門檻 `1e-4`，最低 learning rate 為 `1e-6`。`patience=2` 表示連續 3 個 epoch 沒有達到改善門檻時減半；調整後的值從下一個 epoch 開始使用。所有模型共用相同規則，但實際下降時機由各自的 validation loss 決定。

Notebook 的設定區可以調整 scheduler，或以 `USE_LR_SCHEDULER=False` 執行固定 learning rate 比較；程式介面對應 `ExperimentOptions(use_lr_scheduler=False)`。訓練日誌與 `histories.json` 記錄 `learning_rate`（本 epoch 使用值）及 `next_learning_rate`（scheduler 調整後值），`results.csv` 記錄初始與最終 learning rate。最佳 checkpoint 仍依最低 validation MSE 選擇，完整訓練指定 epochs。

加入 scheduler 前的 checkpoint 會自動判定為不相容並重新訓練。同步新版程式到遠端後，請重新啟動 Notebook kernel 並從頭執行；既有 Notebook 輸出不代表 scheduler 的結果。

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
