"""
Simulation environment for E.D.D.A.I. testing and demonstration.

Provides virtual ecosystems where the AI can learn and adapt.
"""

from .virtual_forest import VirtualForest
from .disturbance_events import DisturbanceEvent, DroughtEvent, FireEvent, FloodEvent

__all__ = ["VirtualForest", "DisturbanceEvent", "DroughtEvent", "FireEvent", "FloodEvent"]
