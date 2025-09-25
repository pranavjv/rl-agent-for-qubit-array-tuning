"""
CapacitanceModel package for quantum device capacitance prediction.

This package provides neural network-based capacitance prediction with uncertainty
estimation and Bayesian inference for plunger gate virtualization.
"""

from .CapacitancePrediction import CapacitancePredictionModel, create_model, create_loss_function
from .BayesianUpdater import CapacitancePredictor
from .KrigingUpdater import InterpolatedCapacitancePredictor

__all__ = [
    'CapacitancePredictionModel',
    'create_model', 
    'create_loss_function',
    'CapacitancePredictor',
    'InterpolatedCapacitancePredictor'
]