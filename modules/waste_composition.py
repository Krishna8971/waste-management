"""
Waste Composition Module
Provides waste composition data based on zone characteristics
"""


class WasteComposition:
    """Manage waste composition data by zone"""
    
    # Research-based waste composition for Indian cities
    ZONE_COMPOSITIONS = {
        'EAST': {
            'description': 'Residential and mixed-use areas',
            'organic_waste': 0.60,
            'recyclable_paper': 0.07,
            'recyclable_plastic': 0.08,
            'recyclable_metal': 0.02,
            'recyclable_glass': 0.02,
            'inert_waste': 0.15,
            'other': 0.06
        },
        'WEST': {
            'description': 'Commercial and educational areas',
            'organic_waste': 0.50,
            'recyclable_paper': 0.12,
            'recyclable_plastic': 0.05,
            'recyclable_metal': 0.02,
            'recyclable_glass': 0.02,
            'inert_waste': 0.15,
            'other': 0.14
        },
        'SOUTH': {
            'description': 'Mixed residential and commercial',
            'organic_waste': 0.58,
            'recyclable_paper': 0.08,
            'recyclable_plastic': 0.07,
            'recyclable_metal': 0.02,
            'recyclable_glass': 0.02,
            'inert_waste': 0.15,
            'other': 0.08
        },
        'MAHADEVPURA': {
            'description': 'IT corridor and tech parks',
            'organic_waste': 0.45,
            'recyclable_paper': 0.15,
            'recyclable_plastic': 0.10,
            'recyclable_metal': 0.03,
            'recyclable_glass': 0.02,
            'inert_waste': 0.15,
            'other': 0.10
        },
        'DASARAHALLI': {
            'description': 'Industrial areas',
            'organic_waste': 0.50,
            'recyclable_paper': 0.08,
            'recyclable_plastic': 0.06,
            'recyclable_metal': 0.04,
            'recyclable_glass': 0.02,
            'inert_waste': 0.20,
            'other': 0.10
        },
        'RR_NAGAR': {
            'description': 'Mixed residential and commercial',
            'organic_waste': 0.55,
            'recyclable_paper': 0.09,
            'recyclable_plastic': 0.06,
            'recyclable_metal': 0.02,
            'recyclable_glass': 0.02,
            'inert_waste': 0.15,
            'other': 0.11
        },
        'YALAHANKA': {
            'description': 'Residential and airport area',
            'organic_waste': 0.60,
            'recyclable_paper': 0.06,
            'recyclable_plastic': 0.05,
            'recyclable_metal': 0.02,
            'recyclable_glass': 0.02,
            'inert_waste': 0.15,
            'other': 0.10
        }
    }
    
    DEFAULT_COMPOSITION = {
        'description': 'Standard urban area',
        'organic_waste': 0.55,
        'recyclable_paper': 0.08,
        'recyclable_plastic': 0.06,
        'recyclable_metal': 0.02,
        'recyclable_glass': 0.02,
        'inert_waste': 0.15,
        'other': 0.12
    }
    
    @classmethod
    def get_composition(cls, zone_name):
        """
        Get waste composition based on zone characteristics
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            dict: Waste composition percentages
        """
        if zone_name in cls.ZONE_COMPOSITIONS:
            return cls.ZONE_COMPOSITIONS[zone_name]
        return cls.DEFAULT_COMPOSITION
    
    @classmethod
    def get_recyclable_fraction(cls, composition):
        """
        Calculate total recyclable fraction from composition
        
        Args:
            composition: Waste composition dictionary
            
        Returns:
            float: Total recyclable fraction
        """
        return (
            composition.get('recyclable_paper', 0) +
            composition.get('recyclable_plastic', 0) +
            composition.get('recyclable_metal', 0) +
            composition.get('recyclable_glass', 0)
        )
    
    @classmethod
    def get_city_average_composition(cls):
        """
        Get city-wide average waste composition
        
        Returns:
            dict: Average waste composition across all zones
        """
        compositions = list(cls.ZONE_COMPOSITIONS.values())
        num_zones = len(compositions)
        
        avg_composition = {'description': 'City-wide average'}
        
        # Calculate average for each waste type
        waste_types = ['organic_waste', 'recyclable_paper', 'recyclable_plastic',
                       'recyclable_metal', 'recyclable_glass', 'inert_waste', 'other']
        
        for waste_type in waste_types:
            total = sum(comp.get(waste_type, 0) for comp in compositions)
            avg_composition[waste_type] = total / num_zones
        
        return avg_composition
    
    @classmethod
    def print_composition(cls, composition):
        """Print waste composition in a formatted way"""
        print(f"\nWaste Composition - {composition.get('description', 'Unknown')}:")
        for waste_type, percentage in composition.items():
            if waste_type != 'description':
                print(f"  {waste_type.replace('_', ' ').title()}: {percentage*100:.1f}%")
