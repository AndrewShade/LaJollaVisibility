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

## Project Structure

- `ocean_data_generator.py`: Handles API connections, scraping, and dataframe merging.
- `unified_ocean_model.py`: Handles model training, evaluation, and plotting.
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