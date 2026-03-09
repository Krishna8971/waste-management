"""
Data Loader Module
Handles loading all data sources for waste analysis
"""

import pandas as pd
import os


class DataLoader:
    """Load and manage all data sources for waste analysis"""
    
    def __init__(self, data_dir=None):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the data files. Defaults to parent directory.
        """
        if data_dir is None:
            # Default to parent directory of the waste_analysis folder
            self.data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.data_dir = data_dir
            
        self.waste_data = None
        self.demographic_data = None
        self.segregation_data = None
        
    def load_all(self):
        """Load all available data sources"""
        print("=== Loading Data Sources ===")
        
        success = self._load_waste_data()
        self._load_demographic_data()
        self._load_segregation_data()
        
        return success
    
    def _load_waste_data(self):
        """Load the main waste generation data"""
        try:
            filepath = os.path.join(
                self.data_dir, 
                "Solid_Waste_Generated_Collected_Processed_Data_Bengaluru_from_2015-16to2017-18_1.csv"
            )
            self.waste_data = pd.read_csv(filepath)
            print(f"✓ Waste data: {self.waste_data.shape[0]} wards")
            return True
        except Exception as e:
            print(f"✗ Error loading waste data: {e}")
            return False
    
    def _load_demographic_data(self):
        """Load demographic data"""
        try:
            filepath = os.path.join(self.data_dir, "bangalore-district-demographic-profile.csv")
            self.demographic_data = pd.read_csv(filepath)
            print(f"✓ Demographic data: {self.demographic_data.shape[0]} records")
            return True
        except Exception as e:
            print(f"✗ Error loading demographic data: {e}")
            return False
    
    def _load_segregation_data(self):
        """Load waste segregation data"""
        try:
            filepath = os.path.join(self.data_dir, "Segeration of waste collected.csv")
            self.segregation_data = pd.read_csv(filepath)
            print(f"✓ Segregation data: {self.segregation_data.shape[0]} records")
            return True
        except Exception as e:
            print(f"✗ Error loading segregation data: {e}")
            return False
    
    def get_ward_names(self):
        """Get list of all available ward names"""
        if self.waste_data is not None:
            return self.waste_data['Ward_Name'].unique().tolist()
        return []
    
    def get_zone_names(self):
        """Get list of all available zone names"""
        if self.waste_data is not None:
            return self.waste_data['Zone_Name'].unique().tolist()
        return []
    
    def get_ward_data(self, ward_name):
        """Get data for a specific ward"""
        if self.waste_data is not None:
            return self.waste_data[self.waste_data['Ward_Name'] == ward_name]
        return None
    
    def get_all_wards_data(self):
        """Get data for all wards"""
        return self.waste_data
