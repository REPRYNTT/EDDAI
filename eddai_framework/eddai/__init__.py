"""
E.D.D.A.I. - Environmental Data-Driven Adaptive Intelligence Framework

This framework implements a novel AI paradigm that treats the environment as its primary
training substrate, co-evolutionary partner, and performance evaluator.

Core Components:
- EnvironmentalSensorium: Multimodal sensory data ingestion
- PlasticNeuralFabric: Self-rewiring neural architectures
- EpigeneticPlaceMemory: Contextual spatiotemporal embeddings
- SymbioticObjectiveFunction: Multi-species utility optimization
- EDDAI: Main orchestrating class
"""

from .eddai import EDDAI
from .sensorium import EnvironmentalSensorium
from .neural_fabric import PlasticNeuralFabric
from .place_memory import EpigeneticPlaceMemory
from .sof import SymbioticObjectiveFunction

__version__ = "0.1.0"
__all__ = [
    "EDDAI",
    "EnvironmentalSensorium",
    "PlasticNeuralFabric",
    "EpigeneticPlaceMemory",
    "SymbioticObjectiveFunction"
]
