# Waste Analysis Package

Waste and biodegradable energy forecasting pipeline for Bengaluru, including:
- historical trend analysis
- future MSW projections
- biogas-to-electricity sensitivity analysis
- optional AI validation
- multi-model forecasting comparison

## Project Structure

```
waste_analysis/
├── __init__.py
├── main.py
├── README.md
├── readme.txt
├── requirements.txt
├── graphs/
└── modules/
    ├── __init__.py
    ├── ai_validator.py
    ├── data_loader.py
    ├── energy_calculator.py
    ├── energy_config.py
    ├── model_comparison.py
    ├── prediction.py
    ├── recycling_analysis.py
    ├── ward_analysis.py
    └── waste_composition.py
```

## What The Pipeline Does

1. Loads annual ML data from `Data/ml_data.csv`.
2. Falls back to monthly data file if annual file is unavailable.
3. Computes historical growth and predicts waste for 2026, 2028, 2030, and 2035.
4. Converts organic waste to biogas and electricity with sensitivity ranges.
5. Reports household-equivalent supply and avoided CO2.
6. Runs model comparison using time-series aware validation.
7. Optionally asks AI models (OpenRouter) to validate growth assumptions.

## Data Files Expected

Primary (preferred):
- `Data/ml_data.csv`

Fallback (used if primary is missing):
- `Data/bengaluru_msw_monthly_2018_2025_clean.csv`

Note:
- In current code, the `Data` directory is expected one level above this folder (same level as `waste_analysis`).

## Setup Guide

### 1. Prerequisites

- Python 3.10+
- pip

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` currently includes:
- pandas
- numpy
- scikit-learn
- matplotlib

### 4. Optional: Enable AI validation

Create `.env` in this folder (`waste_analysis/.env`) and add:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

If no API key is present, AI validation is skipped and analysis still runs.

## Startup Guide

Run from the parent directory (recommended):

```bash
python -m waste_analysis.main
```

Or run directly from this directory:

```bash
python main.py
```

## Energy Sensitivity Configuration

Energy sensitivity is configured in `modules/energy_config.py`:

- Biogas yield (m3/tonne): 80, 100, 120
- Electrical efficiency: 0.25, 0.35, 0.45
- Biogas energy content: 6.0 kWh/m3
- Household consumption: 1200 kWh/year

The report now prints ranges from low-to-high scenarios, for example:
- Electricity potential: `45-78 GWh/year`

## Output Sections

The main run prints:
- historical waste data summary
- growth analysis and AI validation status
- waste generation forecasts
- current biodegradable energy potential (range)
- future biodegradable energy potential (range)
- 2030 summary with min-max values
- model comparison metrics and recommendation

## Module Overview

- `main.py`: end-to-end CLI analysis flow and reporting
- `modules/energy_config.py`: sensitivity ranges and core constants for biogas-electricity conversion
- `modules/model_comparison.py`: baseline + ML model benchmarking with `TimeSeriesSplit`
- `modules/ai_validator.py`: OpenRouter-based validation helpers
- `modules/data_loader.py`: legacy ward/demographic loaders
- `modules/ward_analysis.py`: ward-level historical analysis helpers
- `modules/waste_composition.py`: zone composition assumptions
- `modules/recycling_analysis.py`: recycling and energy analysis helpers
- `modules/energy_calculator.py`: composition-based energy calculator used by recycling analysis utilities
- `modules/prediction.py`: alternative prediction utility based on ward aggregates

## Troubleshooting

- `Data files not found`:
  Ensure `Data/ml_data.csv` exists at the expected parent-level `Data` directory.
- `ModuleNotFoundError` when running direct script:
  Prefer `python -m waste_analysis.main` from the parent directory.
- AI validation unavailable:
  Check `.env` key and internet connectivity. The run still proceeds without AI.
- Model comparison skipped:
  Install/verify `scikit-learn` and `matplotlib`.

## Quick Start (PowerShell)

```powershell
# from .../Research INFO/waste_analysis
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
python main.py
```
