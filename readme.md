# La Jolla Underwater Visibility Predictor

A machine learning pipeline designed to predict underwater visibility conditions for scuba diving in La Jolla, California. The project aggregates real-time oceanographic data (waves, wind, tides, rain) and correlates it with historical dive reports to train an XGBoost model.

The system supports binary classification (Go/No-Go), multi-class classification (Poor/Fair/Good/Excellent), and continuous regression (visibility in feet).

## Pipeline Overview

The pipeline is structured as three sequential notebooks, mirroring a Databricks-style ML workflow:

| Step | Notebook | Description |
| :--- | :--- | :--- |
| 1 | `Notebooks/Feature Engineering.ipynb` | Fetches raw data from all APIs, merges sources, engineers features, and writes the training parquet. |
| 2 | `Notebooks/Model Training.ipynb` | Loads the parquet, tunes an XGBoost model with Bayesian optimization, evaluates performance, and saves the model to JSON. |
| 3 | `Notebooks/Model Inference.ipynb` | Fetches live forecast data, aligns it to the trained model's feature schema, and outputs a predicted dive schedule. |

## Features

### Feature Engineering (`OceanDataGenerator`)
- **Buoy Data**: Fetches historical and real-time wave physics (height, period, energy) from CDIP Station 201 (Scripps Nearshore).
- **Meteorology**: Pulls wind and tide data from NOAA Station 9410230 (Scripps Pier) and rain accumulation from OpenWeatherMap.
- **Ground Truth**: Scrapes and parses historical dive reports to label the dataset.
- **Feature Engineering**:
  - Circular transformation of directional variables (Sine/Cosine) to preserve continuity.
  - 72-hour weighted rain accumulation to model runoff lag.
  - Seasonality encoding using Day-of-Year cyclics.
  - Lag features (1–3 days) for time-series context.

### Modeling Engine (`UnifiedOceanModel`)
- **Algorithm**: XGBoost (Extreme Gradient Boosting).
- **Optimization**: Integrated Hyperopt for Bayesian hyperparameter tuning.
- **Modes**:
  - **Binary**: Optimizes for Recall to minimize false negatives (safety focus).
  - **Classification**: Predicts categorical conditions (Poor, Fair, Good, Excellent).
  - **Regression**: Predicts exact visibility distance in feet.

### Forecasting & Inference (`OceanForecastGenerator` & `OceanInference`)
- **Live Forecasts**: Integrates the Open-Meteo Marine API for wave simulations, alongside OpenWeatherMap and NOAA APIs, to build a forward-looking feature set.
- **Dynamic Alignment**: Automatically aligns forecast features with the specific XGBoost model architecture to prevent strict ordering errors.
- **Flexible Outputs**: Routes logic to handle Regressors, Binary Classifiers, and Multi-class Classifiers, outputting a clean schedule of predicted conditions.

## Prerequisites

- Python 3.10+ (Developed on 3.12)
- OpenWeatherMap API Key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AndrewShade/LaJollaVisibility.git
   cd LaJollaVisibility
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

This project uses `python-dotenv` to manage secrets. Create a `.env` file in the root directory:

```text
# .env
OCEAN_DATA_PATH=training_data.parquet
BINARY_THRESHOLD=20
OWM_API_KEY=your_actual_openweathermap_api_key
```

| Variable | Description |
| :--- | :--- |
| `OWM_API_KEY` | Your private API Key from OpenWeatherMap. |
| `OCEAN_DATA_PATH` | The filename for the aggregated parquet dataset. |
| `BINARY_THRESHOLD` | Visibility (in feet) that defines a "Go" condition (default: 20). |

## Usage

Run the notebooks in order from the `Notebooks/` directory. Each notebook is self-contained and hands off its output (a parquet file or model JSON) to the next step.

### Step 1 — Feature Engineering

Open `Notebooks/Feature Engineering.ipynb` and run all cells.

Fetches 650 days of buoy, wind, tide, and rain data, scrapes dive report labels, and saves two training parquets to `data/`:
- `visibility_data_reg.parquet` — continuous visibility target for regression
- `visibility_data_class.parquet` — binned 0–3 target for classification

### Step 2 — Model Training

Open `Notebooks/Model Training.ipynb` and run all cells.

Trains three XGBoost variants (regression, 4-class, binary) with 100-iteration Bayesian tuning each. Saves models to `data/`:
- `regression_cove_model.json`
- `fourClass_cove_model.json`
- `binary_cove_model.json`

### Step 3 — Inference

Open `Notebooks/Model Inference.ipynb` and run all cells.

Fetches live wave, weather, and tide forecasts, assembles the feature set, and outputs a predicted dive schedule for the upcoming days.

## Project Structure

```
LaJollaVisibility/
├── Notebooks/
│   ├── Feature Engineering.ipynb   # Step 1: data ingestion & feature engineering
│   ├── Model Training.ipynb        # Step 2: XGBoost training & evaluation
│   └── Model Inference.ipynb       # Step 3: live forecast & predictions
├── data/
│   ├── visibility_data_reg.parquet
│   ├── visibility_data_class.parquet
│   ├── forecast_data.parquet
│   ├── regression_cove_model.json
│   ├── fourClass_cove_model.json
│   └── binary_cove_model.json
├── requirements.txt
├── .env                            # Configuration secrets (excluded from Git)
└── .gitignore
```

## Future Work

- **Automated Updates:** Schedule a GitHub Action to pull new forecasts daily and publish the predicted Go/No-Go conditions for the upcoming week.
- **Continuous Learning:** Build a feedback loop to periodically scrape new dive reports, append them to the training dataset, and retrain the model.

## License

Distributed under the MIT License.
