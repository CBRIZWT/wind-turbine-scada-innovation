# WT-Transformer Fault Prediction (Gearbox Failure Detection)

This repository implements gearbox failure detection for wind turbines using temperature forecasting with a Transformer-based encoder. 

The approach forecasts temperature time series and detects anomalies or precursor patterns that indicate imminent gearbox failures. Starting from a baseline model trained on 2-years of data from the first wind turbine (TW1), next fine-tuning that baseline on half-year data from individual turbines (WT2...WTn), and finally evaluating each turbine model on their respective 1-year testing data.

---

## Project overview

Goal: predict gearbox faults by forecasting temperature-related sensor series and using deviations (residuals).

High-level pipeline:
1. Prepare historical SCADA/time-series sensor data.
2. Train a Transformer encoder-based forecasting model on sliding windows.
3. Use the forecasted temperature vs actual to compute residuals or derived features.
4. Evaluate using forecasting metrics (MSE/MAE) and detection metrics (number of FPs, lead time to failure).

---

## Architecture and approach

- Model: Transformer encoder-style model adapted for time-series forecasting (positional encoding, multi-head self-attention, feed-forward blocks). Can use either a simple encoder or encoder-decoder depending on configuration.
- Inputs: multivariate time-series windows (e.g., temperatures, rotational speed, ambient temperature).
- Output: forecasted horizon for target temperature sensor(s).
- Detection: residual-based thresholding or a supervised classifier trained on forecast error patterns and labeled failure windows.

Notes:
- Use normalization per sensor (e.g., standard scaling) using training set statistics.
- Use sliding windows for both training and inference; ensure no data leakage across time (use strict train/val/test splits by time).

---

## Repository layout

This README assumes the repository follows a typical structure. If your repo differs, replace paths in reproduction steps with the actual paths.

- README.md
- requirements.txt
- **`Src/`** - Source code for data and modeling operations
  - **`data/`** - Data loading and preprocessing functions 
    - loader.py
    - preprocessing.py
  - **`model`** - Model definition and evaluation functions
    - transformer_full.py
    - evaluation.py
- **`Notebooks`** - Pre-training, fine tuning and inference notebooks per turbine
- **`Models`** - Model artifacts per turbine
- **`Plots`** - Residual plots and failure detection

