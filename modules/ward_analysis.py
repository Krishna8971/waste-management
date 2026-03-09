"""
Ward Analysis Module
Handles analysis of individual ward data
"""

import pandas as pd
import numpy as np


class WardAnalyzer:
    """Analyze waste data for individual wards"""
    
    YEARS = ['2015-16', '2016-17', '2017-18']
    
    def __init__(self, data_loader):
        """
        Initialize the ward analyzer
        
        Args:
            data_loader: DataLoader instance with loaded data
        """
        self.data_loader = data_loader
        
    def analyze_ward(self, ward_name, quiet=False):
        """
        Analyze ward-specific data
        
        Args:
            ward_name: Name of the ward to analyze
            quiet: If True, suppress detailed output
            
        Returns:
            dict: Ward analysis results or None if not found
        """
        if not quiet:
            print(f"\n=== Ward Analysis: {ward_name} ===")
        
        ward_data = self.data_loader.get_ward_data(ward_name)
        if ward_data is None or ward_data.empty:
            if not quiet:
                print(f"✗ No data found for ward: {ward_name}")
            return None
        
        waste_generated = []
        waste_collected = []
        waste_processed = []
        
        for year in self.YEARS:
            gen_col = f'Total_quantum_of_MSW_generated_in_the_city_(in_Metric_tonnes)_-_{year}'
            col_col = f'Total_quantum_of_MSW_collected_by_the_ULB_or_private_operator_(in_Metric_tonnes)_-_{year}'
            proc_col = f'Average_quantum_of_MSW_that_is_processed_or_recycled_(in_Metric_tonnes)_-_{year}'
            
            waste_generated.append(ward_data[gen_col].sum())
            waste_collected.append(ward_data[col_col].sum())
            waste_processed.append(ward_data[proc_col].sum())
        
        # Calculate efficiencies
        collection_efficiency = [
            col/gen*100 if gen > 0 else 0 
            for col, gen in zip(waste_collected, waste_generated)
        ]
        processing_efficiency = [
            proc/col*100 if col > 0 else 0 
            for proc, col in zip(waste_processed, waste_collected)
        ]
        
        if not quiet:
            self._print_ward_stats(waste_generated, collection_efficiency, processing_efficiency)
        
        return {
            'ward_name': ward_name,
            'zone': ward_data['Zone_Name'].iloc[0],
            'waste_generated': waste_generated,
            'waste_collected': waste_collected,
            'waste_processed': waste_processed,
            'collection_efficiency': collection_efficiency,
            'processing_efficiency': processing_efficiency
        }
    
    def analyze_all_wards(self, quiet=False):
        """
        Analyze data across all wards
        
        Args:
            quiet: If True, suppress detailed output
        
        Returns:
            dict: Aggregated analysis results for all wards
        """
        if not quiet:
            print(f"\n=== All Wards Analysis ===")
        
        all_data = self.data_loader.get_all_wards_data()
        if all_data is None or all_data.empty:
            print("✗ No ward data available")
            return None
        
        # Aggregate statistics across all wards
        total_generated = []
        total_collected = []
        total_processed = []
        ward_stats = []
        
        for year in self.YEARS:
            gen_col = f'Total_quantum_of_MSW_generated_in_the_city_(in_Metric_tonnes)_-_{year}'
            col_col = f'Total_quantum_of_MSW_collected_by_the_ULB_or_private_operator_(in_Metric_tonnes)_-_{year}'
            proc_col = f'Average_quantum_of_MSW_that_is_processed_or_recycled_(in_Metric_tonnes)_-_{year}'
            
            total_generated.append(all_data[gen_col].sum())
            total_collected.append(all_data[col_col].sum())
            total_processed.append(all_data[proc_col].sum())
        
        # Per-ward statistics for growth rate calculation
        ward_names = all_data['Ward_Name'].unique()
        for ward_name in ward_names:
            ward_data = all_data[all_data['Ward_Name'] == ward_name]
            ward_generated = []
            for year in self.YEARS:
                gen_col = f'Total_quantum_of_MSW_generated_in_the_city_(in_Metric_tonnes)_-_{year}'
                ward_generated.append(ward_data[gen_col].sum())
            
            # Calculate growth rates for this ward
            if ward_generated[0] > 0 and ward_generated[1] > 0:
                growth_1 = (ward_generated[1] - ward_generated[0]) / ward_generated[0]
            else:
                growth_1 = None
                
            if ward_generated[1] > 0 and ward_generated[2] > 0:
                growth_2 = (ward_generated[2] - ward_generated[1]) / ward_generated[1]
            else:
                growth_2 = None
            
            ward_stats.append({
                'ward_name': ward_name,
                'zone': ward_data['Zone_Name'].iloc[0],
                'waste_generated': ward_generated,
                'growth_2015_2016': growth_1,
                'growth_2016_2017': growth_2
            })
        
        # Calculate overall efficiencies
        collection_efficiency = [
            col/gen*100 if gen > 0 else 0 
            for col, gen in zip(total_collected, total_generated)
        ]
        processing_efficiency = [
            proc/col*100 if col > 0 else 0 
            for proc, col in zip(total_processed, total_collected)
        ]
        
        if not quiet:
            print(f"Total wards analyzed: {len(ward_names)}")
            self._print_ward_stats(total_generated, collection_efficiency, processing_efficiency, city_level=True)
        
        return {
            'total_wards': len(ward_names),
            'waste_generated': total_generated,
            'waste_collected': total_collected,
            'waste_processed': total_processed,
            'collection_efficiency': collection_efficiency,
            'processing_efficiency': processing_efficiency,
            'ward_stats': ward_stats
        }
    
    def _print_ward_stats(self, waste_generated, collection_efficiency, processing_efficiency, city_level=False):
        """Print ward statistics"""
        prefix = "City-wide" if city_level else "Ward"
        
        print(f"\n{prefix} Waste Generation (Metric Tonnes):")
        for year, gen in zip(self.YEARS, waste_generated):
            print(f"  {year}: {gen:.2f}")
        
        print(f"\nCollection Efficiency (%):")
        for year, eff in zip(self.YEARS, collection_efficiency):
            print(f"  {year}: {eff:.1f}%")
        
        print(f"\nProcessing Efficiency (%):")
        for year, eff in zip(self.YEARS, processing_efficiency):
            print(f"  {year}: {eff:.1f}%")
