# Multivariate Time Series Forecasting of Energy Consumption in the Steel Industry

A complete time series forecasting pipeline applied to electricity consumption data from a South Korean steel plant. The project covers the full modelling workflow: from exploratory analysis and classical statistical models to a modern transformer-based architecture.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | Steel Industry Energy Consumption (2018, 15-min resolution) |
| **Target variable** | `Usage_kWh` — electricity consumed per interval |
| **Forecast horizon** | 24 hours (96 steps at 15-min intervals) |
| **Train / Test split** | Full year 2018 / Last day (Dec 31, 2018) |
| **Evaluation metrics** | MAPE, MSE |

---

## Repository Structure

```
├── Advanced_ML_D2_2.ipynb   # Main notebook: full pipeline
├── Project_Proposal.ipynb   # Initial project proposal
├── data/
│   └── Steel_industry_data.csv
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset records electricity consumption and related electrical variables from a steel manufacturing plant at **15-minute intervals** throughout 2018. Key features include:

- `Usage_kWh` — target: active energy consumed
- `Lagging_Current_Reactive.Power_kVarh`
- `Leading_Current_Reactive_Power_kVarh`
- `CO2(tCO2)` — CO₂ emissions proxy
- `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor`
- `NSM` — number of seconds from midnight
- `WeekStatus`, `Day_of_week`, `Load_Type`

> **Note:** The test period corresponds to **New Year's Eve**, when the plant is essentially idle (~3 kWh baseline). Periodic small spikes (~5 kWh every 4 hours) are attributed to automated maintenance cycles (thermal furnace preservation, anti-freeze water circulation, safety system checks).

---

## Models

### Baselines
- **Mean** — global historical mean
- **Last Value** — last observed value (naïve)
- **AR(1)** — autoregressive model of order 1 via SARIMAX `(1,0,0)`, refitted every `window=2` steps (every 30 minutes)

### Statistical Models
- **ARMA** — identified via ACF/PACF analysis; confirmed autoregressive structure
- **ARIMA** — best `(p,q)` configuration selected via AIC grid search; residual analysis performed via Q-Q plot and Ljung-Box test
- **SARIMA** — seasonal extension; grid search over `(P,D,Q,S)` parameters *(RAM-intensive)*
- **SARIMAX** — SARIMA with exogenous variables (reactive power, power factor, CO₂, load type)

### Deep Learning
- **Chronos** (`amazon/chronos-t5-small`) — zero-shot transformer-based forecaster; median forecast extracted from the predictive distribution

---

## Results Summary

All models are evaluated on the same test set with MAPE and MSE. Results are compiled in the final `Results discussion` section of the notebook.

| Model | MAPE | MSE |
|-------|------|-----|
| Baseline - Mean | - | - |
| Baseline - Last Value | - | - |
| Baseline - AR(1) | - | - |
| ARIMA | - | - |
| SARIMAX | - | - |
| Chronos | - | - |

*Run the notebook to populate this table with actual values.*

---

## Setup

### Requirements

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `statsmodels` | ARIMA / SARIMAX modelling |
| `chronos-forecasting` | Transformer-based forecasting |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib` | Visualization |
| `scikit-learn` | Metrics (MAPE, MSE) |
| `tqdm`, `ipywidgets` | Progress bars in Jupyter |

### Run

Open `Advanced_ML_D2_2.ipynb` in Jupyter or VS Code and run all cells in order.

---

## Key Findings

- The plant exhibits strong **intra-day periodicity** (~4h cycles during idle periods) likely caused by automated thermal maintenance cycles.
- Seasonal patterns reduce significantly during **holiday periods** (July, August, December), making those periods easier to forecast.
- **Last Value** is a surprisingly strong baseline given the near-stationary nature of the test period.
- **ARIMA residuals** did not pass the Ljung-Box test, indicating unexplained autocorrelation — a known limitation of purely univariate models for industrial data.
- **SARIMAX** benefits from exogenous variables (reactive power, power factor), which are leading indicators of consumption.
- **Chronos** operates zero-shot (no fine-tuning), yet remains competitive thanks to its pre-training on diverse time series corpora.

---

## References

- [Time Series Decomposition in Python](https://medium.com/@heyamit10/time-series-decomposition-in-python-049b72a00ba0)
- [Seasonal Decomposition — MCP Analytics](https://mcpanalytics.ai/whitepapers/whitepaper-seasonal-decomposition)
- [statsmodels SARIMAX documentation](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
