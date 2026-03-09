# Waste Analysis Package

A comprehensive waste generation and **biodegradable energy potential** analysis tool for Bengaluru city.

## Module Structure

```
waste_analysis/
├── __init__.py           # Package initialization
├── main.py               # Main entry point & analysis runner
├── README.md             # This file
└── modules/
    ├── __init__.py
    ├── ai_validator.py       # AI-powered validation via OpenRouter
    ├── data_loader.py        # Data loading and management
    ├── energy_calculator.py  # Energy potential calculations
    ├── model_comparison.py   # ML model comparison (RF, SVR, Exponential)
    ├── prediction.py         # Future waste prediction
    ├── recycling_analysis.py # Recycling and energy analysis
    ├── ward_analysis.py      # Individual ward analysis
    └── waste_composition.py  # Waste composition by zone
```

## Key Features

1. **Historical Data Analysis**: Uses ML-ready data from 2018-2025 (`Data/ml_data.csv`)
2. **AI-Validated Predictions**: Optional AI validation of growth rates via OpenRouter API (free models)
3. **Biodegradable Energy Calculations**: Calculates biogas and electricity generation potential
4. **Future Projections**: Predicts waste generation for 2026, 2028, 2030, and 2035
5. **Environmental Impact**: CO2 emissions avoided and households that can be powered
6. **Model Comparison Study**: Compares Exponential Growth, Random Forest, and SVR models

## How to Run

From the `Research INFO` directory:

```bash
python -m waste_analysis.main
```

Or run directly:

```bash
python waste_analysis/main.py
```

## Configuration

### AI Validation (Optional)

To enable AI-powered validation of growth rate predictions:

1. Create a `.env` file in the `waste_analysis/` directory
2. Add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

The system uses free models from OpenRouter for validation.

## Data Source

The analysis uses `Data/ml_data.csv` which contains:
- Year (2018-2025)
- Average daily MSW in tonnes
- Annual MSW in tonnes
- Waste composition percentages (organic, plastic, paper, wet waste)
- Organic tonnes per year
- Data source references

## Modules Description

### Main Entry (`main.py`)
- Loads ML-ready data from CSV
- Calculates historical growth rates
- Generates future waste predictions
- Computes biodegradable energy potential
- Displays comprehensive analysis results

### AIValidator (`modules/ai_validator.py`)
- Validates growth rate predictions using AI models
- Uses OpenRouter API with free model fallback
- Provides AI-suggested growth rates and explanations

### DataLoader (`modules/data_loader.py`)
- Loads waste, demographic, and segregation data
- Provides methods to access ward and zone data

### EnergyCalculator (`modules/energy_calculator.py`)
- Calculates renewable energy potential from organic waste
- Biogas yield: 100 m³ per tonne of organic waste
- Energy content: 6.0 kWh/m³ biogas
- Electrical efficiency: 35%
- CO2 avoidance calculations (0.82 kg CO2/kWh for Indian grid)

### WastePrediction (`modules/prediction.py`)
- Calculates growth rates from historical data
- Blends city-wide and ward-specific trends
- Accounts for population growth

### ModelComparison (`modules/model_comparison.py`)
- Compares three prediction models:
  - **Exponential Growth**: Formula-based compound growth model
  - **Random Forest**: Ensemble ML model with n_estimators=50, max_depth=3
  - **SVR**: Support Vector Regression with RBF kernel
- Uses Leave-One-Out Cross-Validation (LOOCV) for robust evaluation
- Calculates R², RMSE, MAE, and MAPE metrics
- Generates future predictions from all models for comparison

### WasteComposition (`modules/waste_composition.py`)
- Zone-specific waste composition data
- Research-based percentages for Indian cities

### WardAnalyzer (`modules/ward_analysis.py`)
- Analyzes individual ward data
- Calculates collection and processing efficiencies

### RecyclingAnalyzer (`modules/recycling_analysis.py`)
- Analyzes recycling potential
- Energy potential projections

## Output Metrics

The analysis provides:

| Metric | Description |
|--------|-------------|
| Biogas Production | Million m³/year from organic waste |
| Electrical Energy | GWh/year potential |
| Households Powered | Number of homes that can be supplied |
| CO2 Avoided | Tonnes of emissions prevented annually |

## Energy Calculation Constants

| Parameter | Value | Unit |
|-----------|-------|------|
| Biogas yield | 100 | m³/tonne organic waste |
| Biogas energy content | 6.0 | kWh/m³ |
| Electrical efficiency | 35% | - |
| Household consumption | 1,200 | kWh/year (Indian avg) |
| Grid CO2 intensity | 0.82 | kg CO2/kWh |

## Model Comparison Methodology

### Models Compared

| Model | Type | Description |
|-------|------|-------------|
| Exponential Growth | Statistical | y = base × (1 + r)^n where r is avg growth rate |
| Random Forest | ML Ensemble | 50 trees, max_depth=3 (tuned for small data) |
| SVR | ML Kernel | RBF kernel, C=100, with feature scaling |

### Evaluation Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| R² (R-squared) | Variance explained by model | Close to 1.0 |
| RMSE | Root Mean Square Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |
| MAPE | Mean Absolute Percentage Error | Lower is better |

### Cross-Validation Strategy

Due to limited data (8 years: 2018-2025), **Leave-One-Out Cross-Validation (LOOCV)** is used:
- Each data point is used once as test data
- Remaining 7 points used for training
- Provides most robust evaluation for small datasets
- Avoids overfitting detection issues with traditional train/test splits
