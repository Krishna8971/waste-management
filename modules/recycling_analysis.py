"""
Recycling Analysis Module
Analyzes recycling and renewable energy potential
"""

try:
    from .waste_composition import WasteComposition
    from .energy_calculator import EnergyCalculator
except ImportError:
    from waste_composition import WasteComposition
    from energy_calculator import EnergyCalculator


class RecyclingAnalyzer:
    """Analyze recycling and renewable energy potential"""
    
    def __init__(self):
        """Initialize the recycling analyzer"""
        pass
    
    def analyze(self, ward_info, future_predictions, use_city_average=False):
        """
        Analyze recycling and renewable energy potential
        
        Args:
            ward_info: Ward analysis results
            future_predictions: Dictionary of year -> predicted waste
            use_city_average: Use city-wide average composition instead of zone-specific
            
        Returns:
            dict: Recycling and energy analysis results
        """
        print(f"\n=== Recycling & Renewable Energy Analysis ===")
        
        zone = ward_info.get('zone', 'UNKNOWN')
        
        if use_city_average:
            composition = WasteComposition.get_city_average_composition()
            print(f"Using city-wide average composition")
        else:
            composition = WasteComposition.get_composition(zone)
            print(f"Zone: {zone} - {composition['description']}")
        
        WasteComposition.print_composition(composition)
        
        recyclable_fraction = WasteComposition.get_recyclable_fraction(composition)
        
        # Current analysis
        current_waste = ward_info['waste_generated'][-1]
        recyclable_waste = current_waste * recyclable_fraction
        
        current_energy, current_breakdown = EnergyCalculator.calculate_energy_potential(
            current_waste, composition
        )
        
        print(f"\n--- Current Status (2017-18) ---")
        print(f"  Total waste: {current_waste:.2f} metric tonnes")
        print(f"  Recyclable waste: {recyclable_waste:.2f} metric tonnes ({recyclable_fraction*100:.1f}%)")
        print(f"  Energy potential: {current_energy:.0f} kWh ({current_energy/1000:.1f} MWh)")
        
        households = EnergyCalculator.energy_to_households(current_energy)
        print(f"  Could power ~{households:.0f} households for a month")
        
        # Future analysis
        results = {
            'current': {
                'year': '2017-18',
                'total_waste': current_waste,
                'recyclable_waste': recyclable_waste,
                'energy_potential': current_energy
            },
            'future': {}
        }
        
        print(f"\n--- Future Projections ---")
        for year, waste_amount in future_predictions.items():
            recyclable = waste_amount * recyclable_fraction
            energy_potential, energy_breakdown = EnergyCalculator.calculate_energy_potential(
                waste_amount, composition
            )
            
            households = EnergyCalculator.energy_to_households(energy_potential)
            
            print(f"\n{year}:")
            print(f"  Total waste: {waste_amount:.2f} metric tonnes")
            print(f"  Recyclable: {recyclable:.2f} metric tonnes ({recyclable_fraction*100:.1f}%)")
            print(f"  Energy potential: {energy_potential:.0f} kWh ({energy_potential/1000:.1f} MWh)")
            print(f"  Could power ~{households:.0f} households")
            
            # Energy breakdown
            print(f"  Energy breakdown:")
            for waste_type, energy in energy_breakdown.items():
                if energy > 0:
                    pct = energy/energy_potential*100 if energy_potential > 0 else 0
                    print(f"    - {waste_type.replace('_', ' ').title()}: {energy:.0f} kWh ({pct:.1f}%)")
            
            results['future'][year] = {
                'total_waste': waste_amount,
                'recyclable_waste': recyclable,
                'energy_potential': energy_potential,
                'energy_breakdown': energy_breakdown
            }
        
        return results
    
    def analyze_city_wide(self, all_wards_analysis, future_predictions):
        """
        Analyze city-wide recycling and energy potential
        
        Args:
            all_wards_analysis: City-wide analysis results
            future_predictions: Dictionary of year -> predicted waste
            
        Returns:
            dict: City-wide recycling and energy analysis results
        """
        print(f"\n=== City-Wide Recycling & Energy Analysis ===")
        
        # Use city-wide average composition
        composition = WasteComposition.get_city_average_composition()
        WasteComposition.print_composition(composition)
        
        recyclable_fraction = WasteComposition.get_recyclable_fraction(composition)
        
        # Current analysis
        current_waste = all_wards_analysis['waste_generated'][-1]
        recyclable_waste = current_waste * recyclable_fraction
        
        current_energy, current_breakdown = EnergyCalculator.calculate_energy_potential(
            current_waste, composition
        )
        
        print(f"\n--- Current City-Wide Status (2017-18) ---")
        print(f"  Total waste: {current_waste:.2f} metric tonnes")
        print(f"  Recyclable waste: {recyclable_waste:.2f} metric tonnes ({recyclable_fraction*100:.1f}%)")
        print(f"  Energy potential: {current_energy:.0f} kWh ({current_energy/1000:.1f} MWh)")
        
        households = EnergyCalculator.energy_to_households(current_energy)
        print(f"  Could power ~{households:.0f} households for a month")
        
        # Future projections
        print(f"\n--- Future City-Wide Projections ---")
        results = {'current': current_waste, 'future': {}}
        
        for year, waste_amount in future_predictions.items():
            recyclable = waste_amount * recyclable_fraction
            energy_potential, energy_breakdown = EnergyCalculator.calculate_energy_potential(
                waste_amount, composition
            )
            
            households = EnergyCalculator.energy_to_households(energy_potential)
            
            print(f"\n{year}:")
            print(f"  Total waste: {waste_amount:.2f} metric tonnes")
            print(f"  Recyclable: {recyclable:.2f} metric tonnes")
            print(f"  Energy potential: {energy_potential/1000:.1f} MWh")
            print(f"  Could power ~{households:.0f} households")
            
            results['future'][year] = {
                'total_waste': waste_amount,
                'recyclable_waste': recyclable,
                'energy_potential': energy_potential
            }
        
        return results
