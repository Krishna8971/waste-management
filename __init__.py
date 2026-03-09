"""
Waste Analysis Package
A comprehensive waste generation analysis tool for Bengaluru
Uses a SINGLE overall prediction rate with optional AI validation
"""

from .modules.data_loader import DataLoader
from .modules.ward_analysis import WardAnalyzer
from .modules.waste_composition import WasteComposition
from .modules.energy_calculator import EnergyCalculator
from .modules.prediction import WastePrediction
from .modules.recycling_analysis import RecyclingAnalyzer
from .modules.ai_validator import AIValidator

__all__ = [
    'DataLoader',
    'WardAnalyzer', 
    'WasteComposition',
    'EnergyCalculator',
    'WastePrediction',
    'RecyclingAnalyzer',
    'AIValidator'
]
