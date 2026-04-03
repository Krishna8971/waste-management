"""
Energy configuration for biogas sensitivity analysis.
"""

ENERGY_SENSITIVITY = {
    "biogas_yield": {
        "low": 80.0,
        "mid": 100.0,
        "high": 120.0,
    },
    "efficiency": {
        "low": 0.25,
        "mid": 0.35,
        "high": 0.45,
    },
}

# Energy content of biogas (kWh per m^3)
BIOGAS_ENERGY_CONTENT = 6.0

# Average household electricity consumption (kWh per year)
HOUSEHOLD_CONSUMPTION = 1200
