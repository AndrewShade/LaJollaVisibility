# La Jolla Underwater Visibility Predictor

Predicts how good the diving will be at La Jolla Cove, in the same terms a local dive shop uses: **Poor / Fair / Good / Excellent**, plus a Go/No-Go call (Go = Excellent). Labels come from the [Just Get Wet](https://justgetwet.com/blogs/dive-reports-and-conditions) daily dive reports, which carry a rating graphic ("0-10 Poor", "5-15 Fair", "10-20 Good", "20-35 Excellent") alongside a feet range. Features come from ocean, weather and in-water sensor data.

The main use is an **evening (8 pm) forecast**: run it at night, or first thing the next morning, to predict that day from what was observed through the evening before. The strongest single predictor is how murky the water was yesterday at the Scripps Pier turbidity sensor.

## Pipeline

| Step | Notebook | Description |
| :--- | :--- | :--- |
| 1 | `Notebooks/Feature Engineering.ipynb` | Pulls every data source incrementally into `data/raw/`, scrapes the dive-report labels, builds features, and writes the training parquets. Also refreshes recent observations for the evening run. |
| 2 | `Notebooks/Model Training.ipynb` | Trains the evening models (`EveningModels`, validated on time-ordered folds) and the older multi-day models (`UnifiedOceanModel`). |
| 3 | `Notebooks/Model Inference.ipynb` | Evening forecast (`EveningForecast`) and the multi-day forecast (`OceanForecastGenerator` + `OceanInference`). |
| | `Notebooks/ocean_features.py` | Feature logic shared by training and inference, so both compute features identically. |

## Data sources

- **Buoy waves and SST**: CDIP Station 201 (Scripps Nearshore), historic plus realtime.
- **Wind and tide**: NOAA Station 9410230 (Scripps Pier).
- **Rain**: OpenWeatherMap day summaries (needs an API key).
- **Water clarity sensors**: SCCOOS Scripps Pier automated shore station (ERDDAP): turbidity, chlorophyll and CDOM. Only QC-passed readings are used; CDOM is not QC'd by SCCOOS, so it gets a plausibility range instead.
- **Labels**: the dive shop's rating (from the report's graphic) plus its feet range. Posts that use a generic placeholder graphic carry no rating and are skipped for the classification target.

Pulls are incremental. Each source is cached as parquet in `data/raw/`, and later runs only fetch days after the last cached date (re-fetching the last 3 days, since the newest day is usually partial). The caches are committed so nobody has to backfill. For a full rebuild, call `gen.run(1500, 40, full_refresh=True)` or delete `data/raw/`.

## How the evening model works

- **Observations only.** Inputs are yesterday's pier turbidity, chlorophyll and CDOM, recent wave height, wind, rain, SST and season. Forecasts are not model inputs: there is no archive of past forecasts to learn how much to trust them, and they are noisier than the observations used in training.
- **Go/No-Go and rating.** Two XGBoost models, tuned with Hyperopt: Go (the shop rated the day Excellent), with a probability cutoff chosen from out-of-fold F2, and the four-way rating.
- **Honest validation.** Adjacent days are near-duplicates, so a random split overstates skill. Tuning happens on expanding time folds inside the training window, and reported scores come from later held-out months, per fold.
- **Forecast cautions.** `EveningForecast.predict(forecast_df)` can show plain flags (rain forecast, swell building) beside the prediction. They do not change it, and their thresholds are not validated.
- **Forecast archive.** Every forecast pull is also saved to `data/forecast_log/` with a timestamp, so real forecasts can be used for training later.

Use `Go_Probability` for the Go/No-Go call. The rating model is trained with balanced class weights, so its `P(Excellent)` runs higher than the Go model's probability and the two will not always agree.

Limits to keep in mind: there are only ~575 rated days and about 50 Excellent ones, ratings are subjective and their ranges overlap by design, and Excellent days cluster in winter. The rating is predicted within one class most of the time, but exact-class accuracy is modest.

## Prerequisites

- Python 3.10+ (developed on 3.12)
- OpenWeatherMap API key

## Installation

```bash
git clone https://github.com/AndrewShade/LaJollaVisibility.git
cd LaJollaVisibility
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the repo root:

```text
OCEAN_DATA_PATH=../data/visibility_data_reg.parquet
OWM_API_KEY=your_openweathermap_api_key
```

| Variable | Description |
| :--- | :--- |
| `OWM_API_KEY` | Your private API key from OpenWeatherMap. |
| `OCEAN_DATA_PATH` | Regression training parquet (visibility in feet). |

## Usage

Run the notebooks from the `Notebooks/` directory.

### Step 1: Feature Engineering

Run all cells in `Feature Engineering.ipynb`. It writes to `data/`:
- `visibility_data_class.parquet`: target is the shop's rating (0 Poor, 1 Fair, 2 Good, 3 Excellent), rated days only
- `visibility_data_reg.parquet`: target is visibility in feet (lower bound of the reported range), all days

### Step 2: Model Training

Run the `EveningModels` cells to train the evening models (`data/evening_*.json`). The `UnifiedOceanModel` cells train the older multi-day models: `regression_cove_model.json`, `fourClass_cove_model.json`, `binary_cove_model.json`.

### Step 3: Inference

**Evening forecast (recommended).**
1. In `Feature Engineering.ipynb`, run the last cell (`gen.update_observations()`). It is incremental, so it takes seconds.
2. In `Model Inference.ipynb`, run the forecast cell (optional, for cautions), then the "Evening (8 pm) forecast" cells. After noon it predicts tomorrow, before noon it predicts today.

If the pier sensor is down, the result says so and falls back to a weaker estimate. If the observed caches are stale, it tells you to refresh them.

**Multi-day forecast.** The remaining cells pull forecast waves, wind, rain and tide for the next few days and run the older models. Lag, SST and peak-PSD features cannot be computed from forecast APIs and are set to 0, so treat these as rough, especially beyond tomorrow.

## Project structure

```
LaJollaVisibility/
├── Notebooks/
│   ├── Feature Engineering.ipynb   # Step 1: incremental data pulls, labels, features
│   ├── Model Training.ipynb        # Step 2: evening + multi-day models
│   ├── Model Inference.ipynb       # Step 3: evening + multi-day forecasts
│   └── ocean_features.py           # Feature code shared by training and inference
├── data/
│   ├── raw/                        # Incremental per-source caches (committed)
│   ├── forecast_log/               # Timestamped archive of every forecast pull
│   ├── visibility_data_class.parquet
│   ├── visibility_data_reg.parquet
│   ├── forecast_data.parquet
│   ├── evening_*.json              # Evening models and Go cutoff
│   ├── regression_cove_model.json
│   ├── fourClass_cove_model.json
│   └── binary_cove_model.json
├── requirements.txt
├── .env                            # Secrets (excluded from Git)
└── .gitignore
```

## Future work

- **Scheduled evening run:** move the data classes out of the notebooks so a scheduled job can refresh observations, predict, and publish the result.
- **Learn from the forecast archive:** once `data/forecast_log/` has a few months of real forecasts, test whether forecast inputs add value.
- **Tomorrow-aware multi-day model:** the multi-day path still trains on observed same-day features but predicts from forecasts.

## License

Distributed under the MIT License.
