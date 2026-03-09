"""
AI Validator Module
Uses OpenRouter API to validate waste analysis data and predictions
"""

import os
import json
import urllib.request
import urllib.error


class AIValidator:
    """Validate waste analysis using AI models via OpenRouter API"""
    
    # Free models available on OpenRouter (updated Jan 2026)
    FREE_MODELS = [
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "upstage/solar-pro-3:free",
        "arcee-ai/trinity-large-preview:free",
        "liquid/lfm-2.5-1.2b-instruct:free"
    ]
    
    def __init__(self, api_key=None):
        """
        Initialize the AI validator
        
        Args:
            api_key: OpenRouter API key. If not provided, looks for OPENROUTER_API_KEY env variable
        """
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = self.FREE_MODELS[0]  # Default to first free model
        
    def set_model(self, model_name):
        """Set the AI model to use"""
        self.model = model_name
        
    def _make_request(self, prompt):
        """Make a request to OpenRouter API with automatic fallback to other free models"""
        if not self.api_key:
            return None, "No API key provided. Set OPENROUTER_API_KEY environment variable."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/waste-analysis",
            "X-Title": "Waste Analysis Validator"
        }
        
        # Try each model until one works
        last_error = None
        for model in self.FREE_MODELS:
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in urban waste management and data analysis. Provide concise, data-driven insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            try:
                req = urllib.request.Request(
                    self.base_url,
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return result['choices'][0]['message']['content'], None
                    
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8') if e.fp else str(e)
                last_error = f"HTTP Error {e.code}: {error_body}"
                # Try next model on rate limit or server error
                if e.code in [429, 500, 502, 503]:
                    continue
                return None, last_error
            except urllib.error.URLError as e:
                last_error = f"URL Error: {e.reason}"
                continue
            except Exception as e:
                last_error = f"Error: {str(e)}"
                continue
        
        return None, f"All models unavailable. Last error: {last_error}"
    
    def validate_growth_rate(self, historical_data, calculated_rate):
        """
        Validate the calculated growth rate using AI
        
        Args:
            historical_data: Dict with historical waste data
            calculated_rate: The calculated growth rate
            
        Returns:
            tuple: (validation_result, ai_suggested_rate, explanation)
        """
        prompt = f"""Analyze this waste generation data for Bengaluru city and validate the growth rate:

Historical Data:
- 2015-16: {historical_data.get('2015-16', 'N/A')} metric tonnes
- 2016-17: {historical_data.get('2016-17', 'N/A')} metric tonnes  
- 2017-18: {historical_data.get('2017-18', 'N/A')} metric tonnes

Calculated annual growth rate: {calculated_rate*100:.2f}%

Context:
- Bengaluru population growth: ~2.5% annually
- City has 198 wards
- Data shows decrease in 2017-18 which may be due to data collection changes

Please provide:
1. Is this growth rate reasonable? (YES/NO)
2. Your suggested annual growth rate (as a percentage)
3. Brief explanation (2-3 sentences)

Format your response as:
VALID: YES or NO
SUGGESTED_RATE: X.X%
EXPLANATION: Your explanation here"""

        response, error = self._make_request(prompt)
        
        if error:
            print(f"  AI Validation unavailable: {error}")
            return None, None, error
        
        # Parse the response
        try:
            lines = response.strip().split('\n')
            is_valid = None
            suggested_rate = None
            explanation = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('VALID:'):
                    is_valid = 'YES' in line.upper()
                elif line.startswith('SUGGESTED_RATE:'):
                    rate_str = line.split(':')[1].strip().replace('%', '')
                    suggested_rate = float(rate_str) / 100
                elif line.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip()
            
            return is_valid, suggested_rate, explanation
            
        except Exception as e:
            return None, None, f"Could not parse AI response: {response[:200]}"
    
    def validate_predictions(self, predictions, base_year_waste, growth_rate):
        """
        Validate future predictions using AI
        
        Args:
            predictions: Dict of year -> predicted waste
            base_year_waste: Base year waste amount
            growth_rate: Growth rate used for predictions
            
        Returns:
            str: AI validation feedback
        """
        pred_summary = "\n".join([f"  {year}: {waste:.2f} metric tonnes" for year, waste in predictions.items()])
        
        prompt = f"""Validate these waste generation predictions for Bengaluru city:

Base Year (2017-18): {base_year_waste:.2f} metric tonnes
Annual Growth Rate Used: {growth_rate*100:.2f}%

Predictions:
{pred_summary}

Consider:
- Bengaluru is one of India's fastest growing cities
- Waste generation typically correlates with population and economic growth
- Government initiatives may reduce per-capita waste

Provide a brief assessment (3-4 sentences) on whether these predictions are:
1. Realistic given urban growth trends
2. Accounting for waste reduction efforts
3. Any concerns or recommendations"""

        response, error = self._make_request(prompt)
        
        if error:
            return f"AI validation unavailable: {error}"
        
        return response
    
    def get_recommendations(self, current_waste, recyclable_percentage, energy_potential):
        """
        Get AI recommendations for waste management
        
        Args:
            current_waste: Current waste generation
            recyclable_percentage: Percentage of recyclable waste
            energy_potential: Energy potential in kWh
            
        Returns:
            str: AI recommendations
        """
        prompt = f"""Based on this waste analysis for Bengaluru:

Current Waste: {current_waste:.2f} metric tonnes/year
Recyclable Waste: {recyclable_percentage:.1f}%
Energy Potential: {energy_potential/1000:.1f} MWh

Provide 3-4 specific, actionable recommendations for:
1. Improving recycling rates
2. Maximizing energy recovery
3. Reducing overall waste generation

Keep recommendations practical and relevant to Indian urban context."""

        response, error = self._make_request(prompt)
        
        if error:
            return f"AI recommendations unavailable: {error}"
        
        return response


def test_ai_validator():
    """Test the AI validator"""
    validator = AIValidator()
    
    if not validator.api_key:
        print("No API key found. Set OPENROUTER_API_KEY environment variable to test.")
        print("Example: $env:OPENROUTER_API_KEY = 'your-api-key-here'")
        return
    
    print("Testing AI Validator...")
    
    historical = {
        '2015-16': 1255157.40,
        '2016-17': 1321218.31,
        '2017-18': 832038.59
    }
    
    is_valid, suggested, explanation = validator.validate_growth_rate(historical, 0.025)
    print(f"\nValidation Result: {is_valid}")
    print(f"Suggested Rate: {suggested}")
    print(f"Explanation: {explanation}")


if __name__ == "__main__":
    test_ai_validator()
