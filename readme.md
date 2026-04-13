# La Jolla Underwater Visibility Predictor

A machine learning pipeline designed to predict underwater visibility conditions for scuba diving in La Jolla, California. This project aggregates real-time oceanographic data (waves, wind, tides, rain) and correlates it with historical dive reports to train an XGBoost model.

The system supports binary classification (Go/No-Go), multi-class classification (Poor/Fair/Good/Excellent), and continuous regression (visibility in feet).

## Features

### Data Pipeline (`OceanDataGenerator`)
- **Buoy Data**: Fetches historical and real-time wave physics (height, period, energy) from CDIP Station 201 (Scripps Nearshore).
- **Meteorology**: Pulls wind and tide data from NOAA Station 9410230 (Scripps Pier) and rain accumulation from OpenWeatherMap.
- **Ground Truth**: Scrapes and parses historical dive reports to label the dataset.
- **Feature Engineering**:
  - Circular transformation of directional variables (Sine/Cosine) to preserve continuity.
  - 72-hour weighted rain accumulation to model runoff lag.
  - Seasonality encoding using Day-of-Year cyclics.

### Modeling Engine (`UnifiedOceanModel`)
- **Algorithm**: XGBoost (Extreme Gradient Boosting).
- **Optimization**: Integrated Hyperopt for Bayesian hyperparameter tuning.
- **Modes**:
  - **Binary**: Optimizes for Recall to minimize false negatives (safety focus).
  - **Classification**: Predicts categorical conditions (Poor, Fair, Good, Excellent).
  - **Regression**: Predicts exact visibility distance in feet.

### Forecasting & Inference (`OceanForecastGenerator` & `OceanInference`)
- **Live Forecasts**: Integrates the Open-Meteo Marine API for mathematical wave simulations, alongside OpenWeatherMap and NOAA APIs, to build a forward-looking feature set.
- **Dynamic Alignment**: Automatically aligns forecast features with the specific XGBoost model architecture to prevent strict ordering errors.
- **Flexible Outputs**: Seamlessly routes logic to handle Regressors, Binary Classifiers, and Multi-class Classifiers, outputting a clean schedule of predicted conditions.

## Prerequisites

- Python 3.10+ (Developed on 3.12)
- OpenWeatherMap API Key

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/LaJollaVisibility.git](https://github.com/YourUsername/LaJollaVisibility.git)
   cd LaJollaVisibility
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

This project uses `python-dotenv` to manage secrets. You must create a file named `.env` in the root directory with the following variables:

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

### 1. Data Generation
Run the generator to fetch raw data, merge sources, and save the training set.

```python
from ocean_data_generator import OceanDataGenerator

# Initialize and run pipeline (fetch 650 days of history)
gen = OceanDataGenerator()
df = gen.run(days=650)

# Save to parquet
gen.save_data(df, "training_data.parquet")
```

### 2. Model Training
Train the model using the generated data. You can choose between binary, classification, or regression modes.

```python
from unified_ocean_model import UnifiedOceanModel

# Initialize model in Binary Mode (Go/No-Go)
model = UnifiedOceanModel(binary_mode=True)

# Run training with Bayesian Optimization (50 iterations)
model.run(max_evals=50)

# Save the trained model
model.save("lajolla_viz_model.json")
```

### 3. Forecast Generation
Generate a forward-looking dataset for the upcoming days.

```python
from ocean_forecast_generator import OceanForecastGenerator

# Fetch live forecast data
forecast_gen = OceanForecastGenerator()
forecast_df = forecast_gen.run_forecast()

# Save to parquet in the data folder
forecast_gen.save_forecast(forecast_df)
```

### 4. Running Predictions
Load your saved model and apply it to the new forecast data.

```python
from ocean_inference import OceanInference

# Initialize inference (supports 'regressor', 'binary', or 'classifier')
inference = OceanInference(model_path="lajolla_viz_model.json", model_type="binary")

# Run prediction (e.g., probability threshold of 0.4 for a 'Go')
upcoming_forecast = inference.run_predictions(threshold=0.4)
print(upcoming_forecast)
```

## Future Work

With the core training and live inference pipelines complete, future updates will focus on automation and system longevity:

* **Automated Updates:** Setting up a scheduled cron job or GitHub Action to pull new forecasts daily and output the expected go or no-go conditions for the upcoming week automatically.
* **Continuous Learning:** Creating a feedback loop where new dive reports are periodically scraped and added to the historical training dataset so the model can be retrained and calibrated over time.

## Project Structure

- `ocean_data_generator.py`: Handles historical API connections, scraping, and dataframe merging.
- `unified_ocean_model.py`: Handles model training, evaluation, and tuning.
- `ocean_forecast_generator.py`: Builds the live forecast feature set.
- `ocean_inference.py`: Wraps the trained model and forecast data to generate predictions.
- `requirements.txt`: Python package dependencies.
- `.env`: Configuration secrets (excluded from Git).
- `.gitignore`: Standard exclusion rules.

## Contributing

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License.