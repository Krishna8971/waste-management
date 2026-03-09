"""
Waste Prediction Module
Predicts future waste generation using a SINGLE OVERALL prediction rate
derived from ALL wards data, with optional AI validation
"""

import numpy as np

try:
    from .ai_validator import AIValidator
    AI_AVAILABLE = True
except ImportError:
    try:
        from ai_validator import AIValidator
        AI_AVAILABLE = True
    except ImportError:
        AI_AVAILABLE = False


class WastePrediction:
    """Predict future waste generation using a single overall rate with AI validation"""
    
    # Population growth rate for Bengaluru (annual)
    POPULATION_GROWTH_RATE = 0.025  # 2.5% per year
    
    def __init__(self, ward_analyzer, api_key=None):
        """
        Initialize the waste prediction module
        
        Args:
            ward_analyzer: WardAnalyzer instance with loaded data
            api_key: OpenRouter API key for AI validation (optional)
        """
        self.ward_analyzer = ward_analyzer
        self.overall_growth_rate = None  # Single overall rate for ALL predictions
        self.ai_validator = AIValidator(api_key) if AI_AVAILABLE else None
        self.ai_validated = False
        self.ai_suggested_rate = None
        self.ai_explanation = None
        self.historical_data = {}
        
    def calculate_overall_growth_rate(self, all_wards_analysis, use_ai_validation=True, quiet=False):
        """
        Calculate a SINGLE overall growth rate from ALL wards data
        
        Args:
            all_wards_analysis: Results from WardAnalyzer.analyze_all_wards()
            use_ai_validation: Whether to validate with AI
            quiet: If True, suppress detailed output
            
        Returns:
            float: Single overall annual growth rate for all predictions
        """
        # Get city-wide totals
        total_waste = all_wards_analysis['waste_generated']
        
        self.historical_data = {
            '2015-16': total_waste[0],
            '2016-17': total_waste[1],
            '2017-18': total_waste[2]
        }
        
        # Calculate year-over-year growth rates
        growth_1 = (total_waste[1] - total_waste[0]) / total_waste[0] if total_waste[0] > 0 else 0
        growth_2 = (total_waste[2] - total_waste[1]) / total_waste[1] if total_waste[1] > 0 else 0
        
        # Analyze per-ward growth rates for more robust estimation
        ward_stats = all_wards_analysis.get('ward_stats', [])
        valid_growth_rates = []
        
        for ward in ward_stats:
            # Use 2015-2016 growth as it's more reliable
            if ward['growth_2015_2016'] is not None and -0.5 < ward['growth_2015_2016'] < 1.0:
                valid_growth_rates.append(ward['growth_2015_2016'])
        
        if valid_growth_rates:
            median_ward_growth = np.median(valid_growth_rates)
        else:
            median_ward_growth = 0.03
        
        # Determine overall growth rate
        # Note: 2017-18 data shows a significant drop which appears to be data anomaly
        # Using the more reliable 2015-2016 growth and ward-level median
        
        if growth_2 < -0.2:  # Significant decrease indicates data issue
            # Use combination of per-ward median growth and population growth rate
            base_rate = max(median_ward_growth, self.POPULATION_GROWTH_RATE)
        else:
            # Use average of both periods
            base_rate = (growth_1 + growth_2) / 2
        
        # Apply bounds
        self.overall_growth_rate = max(0.02, min(base_rate, 0.10))
        
        # AI Validation
        if use_ai_validation and self.ai_validator and self.ai_validator.api_key:
            is_valid, suggested_rate, explanation = self.ai_validator.validate_growth_rate(
                self.historical_data, 
                self.overall_growth_rate
            )
            
            if is_valid is not None:
                self.ai_validated = True
                self.ai_suggested_rate = suggested_rate
                self.ai_explanation = explanation
                
                # If AI suggests a different rate, blend it
                if suggested_rate and abs(suggested_rate - self.overall_growth_rate) > 0.01:
                    # Blend: 60% calculated, 40% AI suggested
                    self.overall_growth_rate = self.overall_growth_rate * 0.6 + suggested_rate * 0.4
            else:
                self.ai_explanation = explanation  # Store error message
        
        return self.overall_growth_rate
    
    def predict(self, base_waste, target_years, label="", quiet=False):
        """
        Predict future waste using the SINGLE overall rate
        
        Args:
            base_waste: Base year waste amount (2017-18)
            target_years: List of years to predict
            label: Optional label for the prediction (e.g., ward name or "City-Wide")
            quiet: If True, suppress detailed output
            
        Returns:
            dict: Predictions for each target year
        """
        if self.overall_growth_rate is None:
            raise ValueError("Must call calculate_overall_growth_rate first")
        
        predictions = {}
        
        for year in target_years:
            years_from_2017 = year - 2017
            
            # Apply both population growth and waste generation growth
            population_factor = (1 + self.POPULATION_GROWTH_RATE) ** years_from_2017
            waste_growth_factor = (1 + self.overall_growth_rate) ** years_from_2017
            
            # Combined growth factor
            total_growth_factor = population_factor * waste_growth_factor
            
            # Final prediction
            prediction = base_waste * total_growth_factor
            predictions[year] = prediction
        
        return predictions
    
    def predict_ward(self, ward_info, target_years, quiet=False):
        """
        Predict future waste for a specific ward using the SAME overall rate
        
        Args:
            ward_info: Ward analysis results
            target_years: List of years to predict
            quiet: If True, suppress detailed output
            
        Returns:
            dict: Predictions for each target year
        """
        base_waste = ward_info['waste_generated'][2]  # 2017-18 data
        return self.predict(base_waste, target_years, ward_info['ward_name'], quiet)
    
    def predict_city(self, all_wards_analysis, target_years, quiet=False):
        """
        Predict future waste for the entire city using the overall rate
        
        Args:
            all_wards_analysis: City-wide analysis results
            target_years: List of years to predict
            quiet: If True, suppress detailed output
            
        Returns:
            dict: City-wide predictions for each target year
        """
        base_waste = all_wards_analysis['waste_generated'][2]  # 2017-18 data
        predictions = self.predict(base_waste, target_years, "City-Wide", quiet)
        
        return predictions
    
    def get_rate_info(self):
        """Get information about the calculated rate"""
        return {
            'overall_rate': self.overall_growth_rate,
            'population_rate': self.POPULATION_GROWTH_RATE,
            'ai_validated': self.ai_validated,
            'ai_suggested_rate': self.ai_suggested_rate,
            'ai_explanation': self.ai_explanation,
            'historical_data': self.historical_data
        }
