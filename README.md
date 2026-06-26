# FHNN — Flow Hierarchical Neural Network

Streamflow forecasting on the [CAMELS-US](https://ral.ucar.edu/solutions/products/camels) dataset (531 basins) using a hierarchical encoder–decoder LSTM with label-conditioned decoding.

---

## Project Structure

```
FHNN/
├── config.py                          # All dataset, training, and model hyperparameters
├── MODEL.py / MODEL.ipynb             # All model class definitions
├── LSTM.py                            # Core LSTM building blocks
├── UTILS.py                           # Utility functions (metrics, data loading, ensemble)
├── DATA/
│   └── preprocessData.ipynb           # Preprocessing pipeline (see Data section)
├── MODELS/
│   ├── lstm_hierarchical_enc_with_label_dec.ipynb / .sh   # FHNN (main model)
│   ├── ctlstm.py                      # Baseline: CT-LSTM
│   ├── lstm_ar.ipynb / .sh            # Baseline: AR-LSTM
│   ├── rrformer.ipynb / .sh           # Baseline: RRFormer
│   └── tft.ipynb / .sh                # Baseline: Temporal Fusion Transformer
└── ANALYZE/
    ├── score.ipynb                    # Compute NSE/R² metrics from ensemble predictions
    ├── plot.ipynb                     # Generate spatial maps and comparison figures
    └── dump.ipynb                     # Scratch / exploratory analysis
```

---

## Data Preprocessing

> **Prerequisites:** The CAMELS-US raw data must already be preprocessed into `(sites, timesteps, features)` numpy arrays. This is done as a prior step outside this repo (see the CAMELS preprocessing pipeline).

Once the upstream numpy arrays exist, run `DATA/preprocessData.ipynb` to generate the windowed inputs used for training. Key parameters are set in `config.py`:

| Variable | Description |
|---|---|
| `DATASET` | `camels_us_531` |
| `RAW_DIR` | Path to raw CAMELS data |
| `PREPROCESSED_DIR` | Output path for preprocessed numpy files |
| `train_year` | `1985–1993` |
| `valid_year` | `1993–1995` |
| `test_year` | `1995–2005` |
| `window` | Lookback window = 365 days |
| `context` | Context length = 365 days |
| `forecast` | Forecast horizon = 1 day |

### Features

- **Static (27):** Climate indices, geology, soil, topography, land cover
- **Dynamic (5):** PRCP, SRAD, Tmax, Tmin, Vp
- **Target:** Streamflow (SF)

---

## Main Model — FHNN

**File:** `MODELS/lstm_hierarchical_enc_with_label_dec.ipynb`

The FHNN uses a hierarchical LSTM encoder that encodes both static catchment attributes and dynamic forcings, with a label-conditioned decoder for streamflow prediction. All model class definitions live in `MODEL.py`/`MODEL.ipynb`.

Key hyperparameters (`config.py`):

| Parameter | Value |
|---|---|
| `forward_code_dim` | 255 |
| `latent_code_dim` | 85 |
| `dropout` | 0.4 |
| `batch_size` | 128 |
| `epochs` | 50 |
| `learning_rate` | 1e-3 |
| `inits` | 5 (independent runs) |

---

## Baselines

| Model | File |
|---|---|
| CT-LSTM | `MODELS/ctlstm.py` |
| AR-LSTM | `MODELS/lstm_ar.ipynb` |
| RRFormer | `MODELS/rrformer.ipynb` |
| TFT | `MODELS/tft.ipynb` |

---

## Running a Model

Each model in `MODELS/` has a corresponding `.sh` script. The shell script:
1. Converts the `.ipynb` to a `.py` file using `jupyter nbconvert`
2. Runs the resulting Python script with the configuration from `config.py`

```bash
cd MODELS/
bash lstm_hierarchical_enc_with_label_dec.sh   # Run FHNN
bash lstm_ar.sh                                 # Run AR-LSTM baseline
# etc.
```

---

## Results & Ensemble

Each model is run **5 independent times** (`inits = 5` in `config.py`). Final predictions are ensembled by **bagging** (averaging predictions across the 5 runs). Metrics (NSE, R²) are computed against the ensemble predictions.

Results are stored under `RESULT_DIR` (configured in `config.py`). Analysis notebooks:

- `ANALYZE/score.ipynb` — computes per-basin NSE and R² scores
- `ANALYZE/plot.ipynb` — generates spatial maps and scatter plots

---

## Configuration

All variables are centralized in `config.py`. Edit this file to change dataset paths, train/test splits, model dimensions, or training hyperparameters before running.
