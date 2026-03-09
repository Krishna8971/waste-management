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


# ============================================================
# ENERGY CALCULATION CONSTANTS
# ============================================================
# Biogas yield from organic waste (m³ per tonne of organic waste)
BIOGAS_YIELD_PER_TONNE = 100  # m³/tonne (conservative estimate)

# Energy content of biogas (kWh per m³)
BIOGAS_ENERGY_CONTENT = 6.0  # kWh/m³

# Electrical conversion efficiency for biogas generators
ELECTRICAL_EFFICIENCY = 0.35  # 35%

# Average household electricity consumption (kWh per year)
HOUSEHOLD_CONSUMPTION = 1200  # kWh/year (Indian average)


def load_ml_data():
    """Load the ML-ready dataset"""
    data_path = os.path.join(DATA_DIR, "ml_data.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
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


def calculate_energy_potential(organic_tonnes):
    """
    Calculate biodegradable energy generation potential
    
    Args:
        organic_tonnes: Annual organic waste in tonnes
        
    Returns:
        dict: Energy potential metrics
    """
    # Biogas production
    biogas_volume = organic_tonnes * BIOGAS_YIELD_PER_TONNE  # m³/year
    
    # Total energy in biogas
    total_energy = biogas_volume * BIOGAS_ENERGY_CONTENT  # kWh/year
    
    # Electrical energy (after conversion efficiency)
    electrical_energy = total_energy * ELECTRICAL_EFFICIENCY  # kWh/year
    
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


def run_analysis():
    """Run the complete waste and energy analysis"""
    
    print("=" * 70)
    print("  BENGALURU WASTE & BIODEGRADABLE ENERGY ANALYSIS")
    print("=" * 70)
    
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    # ================================================================
    # Load Data
    # ================================================================
    print("\nLoading data from ml_data.csv...")
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
    current_energy = calculate_energy_potential(current_organic)
    
    future_energy = {}
    for year, pred in predictions.items():
        future_energy[year] = calculate_energy_potential(pred['organic_tonnes'])
    
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
   AI-Suggested Rate: {ai_suggested_rate*100:.2f}% {'(USED)' if ai_validated else ''}
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
   Biogas Production:       {current_energy['biogas_million_m3']:>15.2f} million m3/year
   Electrical Energy:       {current_energy['electrical_gwh']:>15.2f} GWh/year
   Households Powered:      {current_energy['households_powered']:>15,.0f} homes
   CO2 Emissions Avoided:   {current_energy['co2_avoided_tonnes']:>15,.0f} tonnes/year""")
    
    # Future Energy Potential
    print(f"""
FUTURE BIODEGRADABLE ENERGY POTENTIAL
   ─────────────────────────────────────────────────────────────""")
    
    for year in sorted(future_energy.keys()):
        energy = future_energy[year]
        print(f"""   
   {year}:
      - Organic Waste:     {predictions[year]['organic_tonnes']:>12,.0f} tonnes/year
      - Biogas Production: {energy['biogas_million_m3']:>12.2f} million m3/year
      - Electrical Energy: {energy['electrical_gwh']:>12.2f} GWh/year
      - Households Powered:{energy['households_powered']:>12,.0f} homes
      - CO2 Avoided:       {energy['co2_avoided_tonnes']:>12,.0f} tonnes/year""")
    
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
  
    {energy_2030['electrical_gwh']:.1f} GWh of electricity per year
    
    Power for {energy_2030['households_powered']:,.0f} households
    
    Avoid {energy_2030['co2_avoided_tonnes']:,.0f} tonnes of CO2 emissions
    
  Using organic waste from the city's municipal solid waste stream.
  
═══════════════════════════════════════════════════════════════════════
""")
    
    # ================================================================
    # MODEL COMPARISON STUDY
    # ================================================================
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON STUDY")
    print("=" * 70)
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
   Based on Leave-One-Out Cross-Validation:
   
   Best Performing Model: {best_model}
   - R² Score: {best_metrics['R2']:.4f}
   - RMSE: {best_metrics['RMSE']:,.0f} tonnes
   - Mean Absolute Percentage Error: {best_metrics['MAPE']:.2f}%
   
   Note: With only 8 data points (2018-2025), all models have limited
   training data. The exponential growth model may extrapolate better
   for long-term predictions, while ML models capture recent patterns.
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
