# Wind Farm Cluster Forecasting–DMD Correction–Economic Dispatch

This repository contains the code and data for the paper:

> *Integrating Stochastic Wind Farm Cluster Output Forecasting with Dynamic Mode Decomposition Based Residual Correction and Optimized Dispatch for Enhanced Wind Power Integration*

The study builds an integrated **forecasting → DMD residual correction → quantile interval forecasting → economic dispatch** workflow for a cluster of three wind farms, and compares the resulting dispatch cost against a Wasserstein-based distributionally robust optimization (DRO) benchmark.

---

## 1. Repository structure

```
.
├── 4月风电场1电力.ipynb      # Full pipeline for Wind Farm 1 (66 MW)
├── 4月风电场2电力.ipynb      # Full pipeline for Wind Farm 2 (96 MW)
├── 4月风电场3电力.ipynb      # Full pipeline for Wind Farm 3 (99 MW)
├── wind power_66MW.xlsx       # Raw meteorological + power data, Wind Farm 1
├── wind power_96MW.xlsx       # Raw meteorological + power data, Wind Farm 2
├── wind power_99MW.xlsx       # Raw meteorological + power data, Wind Farm 3
├── 电力负荷-数据.xlsx          # Electric load data (Problem A, 10th Teddy Cup)
├── requirements.txt            # Python environment
└── README.md
```

Each notebook implements, for the corresponding wind farm:
1. Data loading and min–max normalization
2. Outlier detection and correction (box-plot + IQR)
3. Multi-feature selection (Pearson correlation + Mutual Information + Random Forest)
4. Deterministic point forecasting (LSTM / TCN / GRU / RNN / iTransformer, optimized by PSO)
5. DMD-based residual correction
6. Quantile interval forecasting
7. Simplified economic dispatch with interval constraints, and comparison with the Wasserstein DRO benchmark

---

## 2. Data

| Dataset | Source | Time range | Granularity |
|---|---|---|---|
| Wind farm power + meteorology (3 farms) | 11th Chinese Energy Economics Academic Innovation Competition for College Students | April 2019 (spring) | 15 min |
| Electric load | Problem A, 10th Teddy Cup Mathematical Modeling Contest | — | — |

Meteorological features: wind speed, wind direction, temperature, relative humidity, atmospheric pressure.

Train / validation / test split: **70% / 20% / 10%**.

---

## 3. Environment

- **Python 3.10**
- **TensorFlow 2.15.0** backend
- Hardware used for the experiments: Intel(R) Core(TM) i7-9750H CPU @ 2.60 GHz, 64 GB RAM

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Hyperparameter settings

The PSO search ranges for each model are listed below (see also Table 8 in the paper).

### 4.1 Forecasting models

| Model | Parameter | Search range |
|---|---|---|
| LSTM | Hidden units | [32, 128] |
| LSTM | Training epochs | [50, 200] |
| LSTM | Learning rate | [1×10⁻⁴, 1×10⁻²] |
| LSTM | Batch size | [16, 64] |
| GRU | Hidden units | [32, 128] |
| GRU | Training epochs | [50, 200] |
| GRU | Learning rate | [1×10⁻⁴, 1×10⁻²] |
| GRU | Batch size | [16, 64] |
| RNN | Hidden units | [16, 64] |
| RNN | Training epochs | [50, 200] |
| RNN | Learning rate | [1×10⁻⁴, 1×10⁻²] |
| RNN | Batch size | [16, 64] |
| TCN | Convolutional filters | [10, 60] |
| TCN | Kernel size | [2, 5] |
| TCN | Training epochs | [30, 100] |
| TCN | Learning rate | [1×10⁻⁴, 1×10⁻³] |
| iTransformer | Embedding dimension | [32, 128] |
| iTransformer | Attention heads | [2, 8] |
| iTransformer | Training epochs | [50, 150] |
| iTransformer | Learning rate | [1×10⁻⁴, 1×10⁻³] |

### 4.2 DMD residual correction

| Parameter | Search range |
|---|---|
| Embedding dimension *m* | [6, 24] |
| Truncation rank *r* | [2, 10] |

DMD is applied as a fixed post-processing strategy: the transition operator and modes are fitted on training-set residuals, *m* and *r* are selected on the validation set and fixed for testing. No future information is used at any correction step.

### 4.3 Dispatch model

- Cost coefficients (Zhongyuan Tianhong): wind 0.016, other renewable 0.033, thermal 0.048, storage discharge 0.05, storage charging 0.288 (CNY/kWh).
- Renewable penetration constraint: ≥ 50% of total generation.
- Wasserstein DRO robustness radius: ε = 0.05.
- **Solver: [请填写实际使用的求解器名称及版本，例如 Gurobi 9.5 / CPLEX 12.10 / scipy.optimize / PuLP + CBC / CVXPY]**
- The dispatch model is a **simplified economic demonstration framework**; full state-of-charge dynamics and round-trip efficiency of energy storage are *not* included.

---

## 5. How to reproduce

1. Clone the repository and install dependencies (`pip install -r requirements.txt`).
2. Open `4月风电场1电力.ipynb` (and similarly for farms 2 and 3) in Jupyter Notebook / JupyterLab.
3. Run the cells in order. Each notebook is self-contained for one wind farm and outputs:
   - Deterministic forecast metrics (MSE, MAE, RMSE, R²) before and after DMD correction
   - Interval forecast metrics (PICP, PIAW, ACE, AIS)
   - Dispatch cost comparison (proposed vs. Wasserstein DRO)
4. Aggregate the three-farm results to reproduce the cluster-level tables and figures in the paper.

---

## 6. Notes and limitations

- The dispatch experiments are conducted within a simplified economic dispatch framework and should not be interpreted as evidence of practical operational reliability under full system constraints.
- The 0.07% cost saving over the Wasserstein DRO benchmark is marginal and is reported only under the tested three-farm, spring-season scenario.
- Code is provided as Jupyter notebooks for reproducibility; a script-based refactoring is planned for future releases.

---

## 7. Citation

If you use this code or data, please cite the corresponding paper.

<!-- [BibTeX entry to be added after publication] -->
#（注：内容由AI生成）
