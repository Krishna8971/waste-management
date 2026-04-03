"""
Waste Analysis - Biodegradable Energy Generation Predictor
============================================================
Predicts waste generation and biodegradable energy potential for Bengaluru.
Uses clean ML-ready data from 2018-2025.

Usage:
    python main.py
"""

import os
import sys
import csv

# Ensure proper path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "Data")
sys.path.insert(0, PARENT_DIR)

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(SCRIPT_DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

load_env_file()

from waste_analysis.modules.ai_validator import AIValidator
from waste_analysis.modules.model_comparison import ModelComparison
from waste_analysis.modules.energy_config import (
    BIOGAS_ENERGY_CONTENT,
    HOUSEHOLD_CONSUMPTION,
    ENERGY_SENSITIVITY,
)


# Default reporting mode uses enhanced metrics from time-series model comparison.
USE_ENHANCED_METRICS_DEFAULT = True


def load_ml_data():
    """Load yearly dataset; fallback to aggregated monthly dataset if needed."""
    data_path = os.path.join(DATA_DIR, "ml_data.csv")

    if os.path.exists(data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'year': int(row['year']),
                    'avg_daily_msw_tonnes': float(row['avg_daily_msw_tonnes']),
                    'annual_msw_tonnes': float(row['annual_msw_tonnes']),
                    'organic_pct': float(row['organic_pct']),
                    'plastic_pct': float(row['plastic_pct']),
                    'paper_pct': float(row['paper_pct']),
                    'wet_waste_pct': float(row['wet_waste_pct']),
                    'organic_tonnes_year': float(row['organic_tonnes_year']),
                    'source': row['primary_source']
                })
        return data

    # Fallback: aggregate monthly dataset to yearly rows expected by this analysis.
    monthly_path = os.path.join(DATA_DIR, "bengaluru_msw_monthly_2018_2025_clean.csv")
    if not os.path.exists(monthly_path):
        print(f"❌ Data files not found: {data_path} and {monthly_path}")
        return None

    yearly = {}
    with open(monthly_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row['year'])
            if year not in yearly:
                yearly[year] = {
                    'year': year,
                    'annual_msw_tonnes': 0.0,
                    'organic_tonnes_year': 0.0,
                    'months': 0,
                    'source': 'Aggregated from bengaluru_msw_monthly_2018_2025_clean.csv'
                }

            monthly_msw = float(row['monthly_msw_tonnes'])
            yearly[year]['annual_msw_tonnes'] += monthly_msw
            yearly[year]['months'] += 1

            if 'organic_tonnes_month' in row and row['organic_tonnes_month']:
                yearly[year]['organic_tonnes_year'] += float(row['organic_tonnes_month'])

    data = []
    for year in sorted(yearly.keys()):
        entry = yearly[year]
        annual_msw = entry['annual_msw_tonnes']
        organic_tonnes = entry['organic_tonnes_year'] if entry['organic_tonnes_year'] > 0 else annual_msw * 0.58
        avg_daily = annual_msw / 365.0
        organic_pct = (organic_tonnes / annual_msw * 100.0) if annual_msw > 0 else 58.0

        data.append({
            'year': year,
            'avg_daily_msw_tonnes': avg_daily,
            'annual_msw_tonnes': annual_msw,
            'organic_pct': organic_pct,
            'plastic_pct': 0.0,
            'paper_pct': 0.0,
            'wet_waste_pct': organic_pct,
            'organic_tonnes_year': organic_tonnes,
            'source': entry['source']
        })

    return data


def calculate_growth_rate(data):
    """Calculate average annual growth rate from historical data"""
    growth_rates = []
    
    for i in range(1, len(data)):
        prev = data[i-1]['annual_msw_tonnes']
        curr = data[i]['annual_msw_tonnes']
        if prev > 0:
            rate = (curr - prev) / prev
            growth_rates.append(rate)
    
    # Return average growth rate
    if growth_rates:
        return sum(growth_rates) / len(growth_rates)
    return 0.05  # Default 5%


def predict_future_waste(data, target_years, growth_rate=None):
    """Predict waste generation for future years"""
    # Use last known data point as base
    base_year = data[-1]['year']
    base_waste = data[-1]['annual_msw_tonnes']
    base_organic_pct = data[-1]['organic_pct'] / 100
    
    # Use provided growth rate or calculate from historical data
    if growth_rate is None:
        growth_rate = calculate_growth_rate(data)
    
    predictions = {}
    for year in target_years:
        years_ahead = year - base_year
        if years_ahead > 0:
            predicted_waste = base_waste * ((1 + growth_rate) ** years_ahead)
            # Organic percentage tends to decrease slightly over time
            organic_pct = max(0.50, base_organic_pct - (years_ahead * 0.005))
            predictions[year] = {
                'total_waste': predicted_waste,
                'organic_pct': organic_pct,
                'organic_tonnes': predicted_waste * organic_pct
            }
    
    return predictions, growth_rate


def calculate_energy_potential(organic_tonnes, biogas_yield_per_tonne, electrical_efficiency):
    """
    Calculate biodegradable energy generation potential for one parameter set.
    
    Args:
        organic_tonnes: Annual organic waste in tonnes
        biogas_yield_per_tonne: Biogas yield in m3 per tonne
        electrical_efficiency: Electrical conversion efficiency [0, 1]
        
    Returns:
        dict: Energy potential metrics
    """
    # Biogas production
    biogas_volume = organic_tonnes * biogas_yield_per_tonne  # m³/year
    
    # Total energy in biogas
    total_energy = biogas_volume * BIOGAS_ENERGY_CONTENT  # kWh/year
    
    # Electrical energy (after conversion efficiency)
    electrical_energy = total_energy * electrical_efficiency  # kWh/year
    
    # Convert to MWh
    electrical_mwh = electrical_energy / 1000
    
    # Households that can be powered
    households_powered = electrical_energy / HOUSEHOLD_CONSUMPTION
    
    # CO2 emissions avoided (0.82 kg CO2 per kWh for Indian grid)
    co2_avoided = electrical_energy * 0.82 / 1000  # tonnes CO2/year
    
    return {
        'biogas_million_m3': biogas_volume / 1_000_000,
        'electrical_mwh': electrical_mwh,
        'electrical_gwh': electrical_mwh / 1000,
        'households_powered': households_powered,
        'co2_avoided_tonnes': co2_avoided
    }


def calculate_energy_sensitivity(organic_tonnes):
    """Run sensitivity analysis across configured biogas and efficiency ranges."""
    scenarios = {}

    for yield_label, biogas_yield in ENERGY_SENSITIVITY['biogas_yield'].items():
        for efficiency_label, efficiency in ENERGY_SENSITIVITY['efficiency'].items():
            scenario_name = f"{yield_label}_{efficiency_label}"
            scenarios[scenario_name] = calculate_energy_potential(
                organic_tonnes,
                biogas_yield,
                efficiency,
            )

    metric_keys = [
        'biogas_million_m3',
        'electrical_mwh',
        'electrical_gwh',
        'households_powered',
        'co2_avoided_tonnes',
    ]
    ranges = {}
    for key in metric_keys:
        values = [scenario[key] for scenario in scenarios.values()]
        ranges[key] = {
            'min': min(values),
            'max': max(values),
        }

    mid_case = calculate_energy_potential(
        organic_tonnes,
        ENERGY_SENSITIVITY['biogas_yield']['mid'],
        ENERGY_SENSITIVITY['efficiency']['mid'],
    )

    return {
        'scenarios': scenarios,
        'ranges': ranges,
        'mid_case': mid_case,
    }


def run_analysis():
    """Run the complete waste and energy analysis"""
    
    print("=" * 70)
    print("  BENGALURU WASTE & BIODEGRADABLE ENERGY ANALYSIS")
    print("=" * 70)
    
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    # ================================================================
    # Load Data
    # ================================================================
    print("\nLoading data (ml_data.csv with monthly fallback)...")
    data = load_ml_data()
    
    if not data:
        return
    
    print(f"   Loaded {len(data)} years of data ({data[0]['year']}-{data[-1]['year']})")
    
    # ================================================================
    # Analyze Historical Data
    # ================================================================
    print("\nAnalyzing historical trends...")
    calculated_rate = calculate_growth_rate(data)
    
    # ================================================================
    # AI Validation
    # ================================================================
    ai_validated = False
    ai_suggested_rate = None
    ai_explanation = None
    final_rate = calculated_rate
    
    if api_key:
        print("\nValidating with AI...")
        ai_validator = AIValidator(api_key)
        
        # Prepare historical data for validation
        historical = {str(d['year']): d['annual_msw_tonnes'] for d in data[-4:]}
        
        is_valid, suggested_rate, explanation = ai_validator.validate_growth_rate(
            historical, calculated_rate
        )
        
        if is_valid is not None:
            ai_validated = True
            ai_suggested_rate = suggested_rate
            ai_explanation = explanation
            
            # Use AI-suggested rate if available
            if suggested_rate is not None:
                final_rate = suggested_rate
                print(f"   Using AI-suggested rate: {final_rate*100:.2f}%")
    
    # ================================================================
    # Generate Predictions (using AI-suggested rate if available)
    # ================================================================
    print("\nGenerating predictions...")
    target_years = [2026, 2028, 2030, 2035]
    predictions, _ = predict_future_waste(data, target_years, growth_rate=final_rate)
    
    # ================================================================
    # Calculate Energy Potential
    # ================================================================
    current_organic = data[-1]['organic_tonnes_year']
    current_energy = calculate_energy_sensitivity(current_organic)
    
    future_energy = {}
    for year, pred in predictions.items():
        future_energy[year] = calculate_energy_sensitivity(pred['organic_tonnes'])
    
    # ================================================================
    # DISPLAY RESULTS
    # ================================================================
    print("\n")
    print("=" * 70)
    print("  ANALYSIS RESULTS")
    print("=" * 70)
    
    # Historical Data Summary
    print(f"""
HISTORICAL WASTE DATA (2018-2025)
   ─────────────────────────────────────────────────────────────""")
    
    for d in data:
        print(f"   {d['year']}: {d['annual_msw_tonnes']:>12,.0f} tonnes/year  ({d['organic_pct']:.0f}% organic)")
    
    print(f"""
GROWTH ANALYSIS
   ─────────────────────────────────────────────────────────────
   Calculated Growth Rate: {calculated_rate*100:.2f}%
    AI-Suggested Rate: {(f'{ai_suggested_rate*100:.2f}%' if ai_suggested_rate is not None else 'N/A')} {'(USED)' if ai_validated and ai_suggested_rate is not None else ''}
   Final Rate Applied: {final_rate*100:.2f}%
   AI Validated: {'Yes' if ai_validated else 'No'}
   Data Source: {data[-1]['source']}""")
    
    # Future Predictions
    print(f"""
WASTE GENERATION PREDICTIONS
   ─────────────────────────────────────────────────────────────""")
    
    base_waste = data[-1]['annual_msw_tonnes']
    for year in sorted(predictions.keys()):
        pred = predictions[year]
        growth = ((pred['total_waste'] / base_waste) - 1) * 100
        print(f"   {year}: {pred['total_waste']:>12,.0f} tonnes/year  ({growth:>+6.1f}% from 2025)")
    
    # Current Energy Potential
    print(f"""
CURRENT BIODEGRADABLE ENERGY POTENTIAL (2025)
   ─────────────────────────────────────────────────────────────
   Organic Waste Available: {current_organic:>15,.0f} tonnes/year
    Biogas Production:       {current_energy['ranges']['biogas_million_m3']['min']:>7.2f}-{current_energy['ranges']['biogas_million_m3']['max']:<7.2f} million m3/year
    Electrical Energy:       {current_energy['ranges']['electrical_gwh']['min']:>7.2f}-{current_energy['ranges']['electrical_gwh']['max']:<7.2f} GWh/year
    Households Powered:      {current_energy['ranges']['households_powered']['min']:>7,.0f}-{current_energy['ranges']['households_powered']['max']:<7,.0f} homes
    CO2 Emissions Avoided:   {current_energy['ranges']['co2_avoided_tonnes']['min']:>7,.0f}-{current_energy['ranges']['co2_avoided_tonnes']['max']:<7,.0f} tonnes/year
    Mid-Case Electricity:    {current_energy['mid_case']['electrical_gwh']:>15.2f} GWh/year""")

    print(f"""
SENSITIVITY PARAMETERS
    ─────────────────────────────────────────────────────────────
    Biogas Yield (m3/tonne): low={ENERGY_SENSITIVITY['biogas_yield']['low']:.0f}, mid={ENERGY_SENSITIVITY['biogas_yield']['mid']:.0f}, high={ENERGY_SENSITIVITY['biogas_yield']['high']:.0f}
    Electrical Efficiency:   low={ENERGY_SENSITIVITY['efficiency']['low']:.2f}, mid={ENERGY_SENSITIVITY['efficiency']['mid']:.2f}, high={ENERGY_SENSITIVITY['efficiency']['high']:.2f}""")
    
    # Future Energy Potential
    print(f"""
FUTURE BIODEGRADABLE ENERGY POTENTIAL
   ─────────────────────────────────────────────────────────────""")
    
    for year in sorted(future_energy.keys()):
        energy = future_energy[year]
        print(f"""   
   {year}:
      - Organic Waste:     {predictions[year]['organic_tonnes']:>12,.0f} tonnes/year
            - Biogas Production: {energy['ranges']['biogas_million_m3']['min']:>5.2f}-{energy['ranges']['biogas_million_m3']['max']:<5.2f} million m3/year
            - Electrical Energy: {energy['ranges']['electrical_gwh']['min']:>5.2f}-{energy['ranges']['electrical_gwh']['max']:<5.2f} GWh/year
            - Households Powered:{energy['ranges']['households_powered']['min']:>5,.0f}-{energy['ranges']['households_powered']['max']:<5,.0f} homes
            - CO2 Avoided:       {energy['ranges']['co2_avoided_tonnes']['min']:>5,.0f}-{energy['ranges']['co2_avoided_tonnes']['max']:<5,.0f} tonnes/year""")
    
    # AI Validation Section
    if ai_validated:
        print(f"""
AI VALIDATION
   ─────────────────────────────────────────────────────────────
   Status: Validated
   AI Suggested Rate: {ai_suggested_rate*100:.2f}% per year
   Explanation: {ai_explanation}""")
    elif api_key:
        print(f"""
AI VALIDATION
   ─────────────────────────────────────────────────────────────
   Status: Could not validate
   Reason: {ai_explanation or 'Unknown error'}""")
    
    # Summary
    energy_2030 = future_energy.get(2030, future_energy.get(max(future_energy.keys())))
    print(f"""
═══════════════════════════════════════════════════════════════════════
  SUMMARY: BIODEGRADABLE ENERGY GENERATION OPPORTUNITY
═══════════════════════════════════════════════════════════════════════

  By 2030, Bengaluru can generate approximately:
  
        {energy_2030['ranges']['electrical_gwh']['min']:.1f}-{energy_2030['ranges']['electrical_gwh']['max']:.1f} GWh of electricity per year
    
        Power for {energy_2030['ranges']['households_powered']['min']:,.0f}-{energy_2030['ranges']['households_powered']['max']:,.0f} households
    
        Avoid {energy_2030['ranges']['co2_avoided_tonnes']['min']:,.0f}-{energy_2030['ranges']['co2_avoided_tonnes']['max']:,.0f} tonnes of CO2 emissions
    
  Using organic waste from the city's municipal solid waste stream.
  
═══════════════════════════════════════════════════════════════════════
""")
    
    # ================================================================
    # MODEL COMPARISON STUDY
    # ================================================================
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON STUDY")
    print("=" * 70)
    if USE_ENHANCED_METRICS_DEFAULT:
        print("\nComparing prediction models with enhanced default metrics: Time Series Split + monthly tonnage + 95% prediction intervals")
    else:
        print("\nComparing prediction models: Exponential Growth vs Random Forest vs SVR")
    
    try:
        comparison = ModelComparison()
        comparison.load_data()
        comparison.run_comparison()
        print(comparison.format_results())
        
        # Recommend best model
        best_model = comparison.get_best_model()
        best_metrics = comparison.results[best_model]
        
        print(f"""
RECOMMENDATION
   ─────────────────────────────────────────────────────────────
    Based on enhanced default metrics (Time Series Split Cross-Validation):
   
   Best Performing Model: {best_model}
   - R² Score: {best_metrics['R2']:.4f}
    - RMSE: {best_metrics['RMSE']:,.0f} tonnes/month
   - Mean Absolute Percentage Error: {best_metrics['MAPE']:.2f}%
   
    Note: Metrics are calculated on monthly data using forward-chaining
    validation, so this is now the default enhanced evaluation path.
""")
        
        # Generate visualization graphs
        print("\nGenerating comparison visualizations...")
        try:
            graph_files = comparison.generate_visualizations()
            print("\n   Generated graphs:")
            for gf in graph_files:
                print(f"      - {os.path.basename(gf)}")
        except Exception as viz_error:
            print(f"\n   ⚠️ Visualization generation failed: {viz_error}")
            print("   Install matplotlib: pip install matplotlib>=3.7.0")
            
    except Exception as e:
        print(f"\n   ⚠️ Model comparison skipped: {e}")
        print("   Install scikit-learn: pip install scikit-learn>=1.3.0")

    print("=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
