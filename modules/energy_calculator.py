"""
Energy Calculator Module
Calculates renewable energy potential from waste
"""


class EnergyCalculator:
    """Calculate renewable energy potential from waste"""
    
    # Energy conversion factors (kWh per kg of waste)
    ENERGY_FACTORS = {
        'organic_waste': 0.5,      # Anaerobic digestion
        'recyclable_paper': 4.0,   # Incineration
        'recyclable_plastic': 8.0, # Incineration
        'recyclable_metal': 0.0,   # No energy recovery
        'recyclable_glass': 0.0,   # No energy recovery
        'inert_waste': 0.0,        # No energy recovery
        'other': 2.0               # Mixed waste incineration
    }
    
    @classmethod
    def calculate_energy_potential(cls, waste_amount_tonnes, composition):
        """
        Calculate renewable energy potential from waste
        
        Args:
            waste_amount_tonnes: Total waste amount in metric tonnes
            composition: Waste composition dictionary
            
        Returns:
            tuple: (total_energy_kwh, energy_breakdown_dict)
        """
        total_energy = 0
        energy_breakdown = {}
        
        for waste_type, percentage in composition.items():
            if waste_type == 'description':
                continue
            
            # Convert tonnes to kg
            waste_kg = waste_amount_tonnes * 1000 * percentage
            energy = waste_kg * cls.ENERGY_FACTORS.get(waste_type, 0)
            total_energy += energy
            energy_breakdown[waste_type] = energy
        
        return total_energy, energy_breakdown
    
    @classmethod
    def energy_to_households(cls, energy_kwh):
        """
        Convert energy to equivalent households powered
        
        Args:
            energy_kwh: Energy in kilowatt-hours
            
        Returns:
            float: Number of households that could be powered
        """
        # Average household consumption in India: ~100 kWh/month
        monthly_consumption = 100
        return energy_kwh / monthly_consumption
    
    @classmethod
    def print_energy_report(cls, waste_amount, composition, year=None):
        """
        Print a detailed energy report
        
        Args:
            waste_amount: Waste amount in metric tonnes
            composition: Waste composition dictionary
            year: Optional year label
        """
        total_energy, breakdown = cls.calculate_energy_potential(waste_amount, composition)
        
        year_label = f" ({year})" if year else ""
        print(f"\nEnergy Potential{year_label}:")
        print(f"  Total waste: {waste_amount:.2f} metric tonnes")
        print(f"  Total energy potential: {total_energy:.0f} kWh ({total_energy/1000:.1f} MWh)")
        
        households = cls.energy_to_households(total_energy)
        print(f"  Could power ~{households:.0f} households for a month")
        
        print(f"\n  Energy breakdown:")
        for waste_type, energy in breakdown.items():
            if energy > 0:
                print(f"    - {waste_type.replace('_', ' ').title()}: {energy:.0f} kWh ({energy/total_energy*100:.1f}%)")
        
        return total_energy, breakdown
