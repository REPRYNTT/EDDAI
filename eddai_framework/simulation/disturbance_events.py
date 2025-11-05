"""
Disturbance Events - Simulated environmental disturbances for testing E.D.D.A.I. adaptation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class DisturbanceEvent(ABC):
    """
    Base class for environmental disturbance events.
    """

    def __init__(self, intensity: float = 1.0, duration_days: float = 7.0):
        """
        Initialize disturbance event.

        Args:
            intensity: Severity of the disturbance (0-1)
            duration_days: How long the disturbance lasts
        """
        self.intensity = intensity
        self.duration_days = duration_days
        self.elapsed_days = 0.0
        self.active = True

    @abstractmethod
    def update(self, dt_days: float, environment: np.ndarray, weather: Dict[str, float]) -> bool:
        """
        Update the disturbance effect.

        Args:
            dt_days: Time step in days
            environment: Current environment state
            weather: Current weather conditions

        Returns:
            True if disturbance should continue, False if finished
        """
        self.elapsed_days += dt_days

        if self.elapsed_days >= self.duration_days:
            self.active = False
            return False

        return True

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the disturbance."""
        pass


class DroughtEvent(DisturbanceEvent):
    """
    Drought disturbance - reduces water availability and affects plant growth.
    """

    def __init__(self, intensity: float = 1.0, duration_days: float = 14.0):
        super().__init__(intensity, duration_days)

    def update(self, dt_days: float, environment: np.ndarray, weather: Dict[str, float]) -> bool:
        if not super().update(dt_days, environment, weather):
            return False

        # Reduce water more severely during drought
        progress = self.elapsed_days / self.duration_days
        drought_factor = 1 + self.intensity * (1 - progress) * 2  # Peak at start

        # Water evaporation increases
        water_loss = 0.1 * self.intensity * dt_days * drought_factor
        environment[:, :, 1] -= water_loss
        environment[:, :, 1] = np.maximum(environment[:, :, 1], 0.05)  # Minimum water level

        # Trees suffer more
        tree_stress = 0.05 * self.intensity * dt_days * drought_factor
        environment[:, :, 0] -= tree_stress * (1 - environment[:, :, 1])  # Worse in dry areas
        environment[:, :, 0] = np.maximum(environment[:, :, 0], 0.0)

        # Pollinators migrate away
        pollinator_loss = 0.03 * self.intensity * dt_days * drought_factor
        environment[:, :, 2] -= pollinator_loss
        environment[:, :, 2] = np.maximum(environment[:, :, 2], 0.0)

        # Carbon emission increases due to stress
        carbon_stress = 0.02 * self.intensity * dt_days * drought_factor
        environment[:, :, 3] -= carbon_stress
        environment[:, :, 3] = np.maximum(environment[:, :, 3], 0.0)

        return True

    def get_description(self) -> str:
        return f"Drought (intensity: {self.intensity:.1f}, {self.elapsed_days:.1f}/{self.duration_days:.1f} days)"


class FireEvent(DisturbanceEvent):
    """
    Wildfire disturbance - destroys vegetation and affects soil conditions.
    """

    def __init__(self, intensity: float = 1.0, duration_days: float = 3.0, burn_area: float = 0.3):
        super().__init__(intensity, duration_days)
        self.burn_area = burn_area  # Fraction of area affected
        self.burn_mask = None

    def update(self, dt_days: float, environment: np.ndarray, weather: Dict[str, float]) -> bool:
        if not super().update(dt_days, environment, weather):
            return False

        # Initialize burn area on first update
        if self.burn_mask is None:
            rows, cols = environment.shape[:2]
            total_cells = rows * cols
            burn_cells = int(total_cells * self.burn_area)

            # Create a contiguous burn area (simplified wildfire spread)
            center_row, center_col = rows // 2, cols // 2
            distances = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    distances[i, j] = np.sqrt((i - center_row)**2 + (j - center_col)**2)

            # Select closest cells for burning
            flat_distances = distances.flatten()
            burn_indices = np.argsort(flat_distances)[:burn_cells]
            self.burn_mask = np.zeros((rows, cols), dtype=bool)
            self.burn_mask.flat[burn_indices] = True

        # Apply fire effects
        progress = self.elapsed_days / self.duration_days
        fire_intensity_factor = np.sin(progress * np.pi)  # Peak in middle

        # Trees burn
        tree_burn = 0.8 * self.intensity * dt_days * fire_intensity_factor
        environment[self.burn_mask, 0] -= tree_burn
        environment[:, :, 0] = np.maximum(environment[:, :, 0], 0.0)

        # Water increases temporarily from firefighting/suppression
        if progress < 0.5:  # Early firefighting
            water_boost = 0.1 * self.intensity * dt_days
            environment[self.burn_mask, 1] += water_boost
            environment[:, :, 1] = np.minimum(environment[:, :, 1], 1.0)

        # Pollinators flee
        pollinator_flee = 0.9 * self.intensity * dt_days * fire_intensity_factor
        environment[self.burn_mask, 2] -= pollinator_flee
        environment[:, :, 2] = np.maximum(environment[:, :, 2], 0.0)

        # Carbon releases spike
        carbon_release = 0.5 * self.intensity * dt_days * fire_intensity_factor
        environment[self.burn_mask, 3] -= carbon_release
        environment[:, :, 3] = np.maximum(environment[:, :, 3], 0.0)

        return True

    def get_description(self) -> str:
        burn_percent = self.burn_area * 100
        return f"Wildfire (intensity: {self.intensity:.1f}, burn area: {burn_percent:.1f}%, {self.elapsed_days:.1f}/{self.duration_days:.1f} days)"


class FloodEvent(DisturbanceEvent):
    """
    Flood disturbance - increases water and affects soil stability.
    """

    def __init__(self, intensity: float = 1.0, duration_days: float = 5.0):
        super().__init__(intensity, duration_days)

    def update(self, dt_days: float, environment: np.ndarray, weather: Dict[str, float]) -> bool:
        if not super().update(dt_days, environment, weather):
            return False

        progress = self.elapsed_days / self.duration_days
        flood_factor = np.sin(progress * np.pi)  # Peak in middle

        # Water levels surge
        water_surge = 0.4 * self.intensity * dt_days * flood_factor
        environment[:, :, 1] += water_surge
        environment[:, :, 1] = np.minimum(environment[:, :, 1], 1.0)

        # Trees may be damaged by flooding
        tree_damage = 0.1 * self.intensity * dt_days * flood_factor * (environment[:, :, 1] > 0.8)
        environment[:, :, 0] -= tree_damage
        environment[:, :, 0] = np.maximum(environment[:, :, 0], 0.0)

        # Pollinators may benefit from increased water or be washed away
        if progress < 0.7:  # During flood
            pollinator_change = -0.05 * self.intensity * dt_days * flood_factor  # Mostly negative
        else:  # After flood
            pollinator_change = 0.02 * self.intensity * dt_days  # Some recovery

        environment[:, :, 2] += pollinator_change
        environment[:, :, 2] = np.clip(environment[:, :, 2], 0.0, 1.0)

        # Carbon effects: erosion vs deposition
        if progress < 0.5:
            carbon_erosion = 0.1 * self.intensity * dt_days * flood_factor
            environment[:, :, 3] -= carbon_erosion
        else:
            carbon_deposition = 0.05 * self.intensity * dt_days
            environment[:, :, 3] += carbon_deposition

        environment[:, :, 3] = np.clip(environment[:, :, 3], 0.0, 1.0)

        return True

    def get_description(self) -> str:
        return f"Flood (intensity: {self.intensity:.1f}, {self.elapsed_days:.1f}/{self.duration_days:.1f} days)"


class InvasiveSpeciesEvent(DisturbanceEvent):
    """
    Invasive species disturbance - competes with native species.
    """

    def __init__(self, intensity: float = 1.0, duration_days: float = 30.0, species_type: str = "herbivore"):
        super().__init__(intensity, duration_days)
        self.species_type = species_type

    def update(self, dt_days: float, environment: np.ndarray, weather: Dict[str, float]) -> bool:
        if not super().update(dt_days, environment, weather):
            return False

        progress = min(1.0, self.elapsed_days / self.duration_days)

        if self.species_type == "herbivore":
            # Herbivores eat trees and compete with pollinators
            tree_consumption = 0.05 * self.intensity * dt_days * progress
            environment[:, :, 0] -= tree_consumption
            environment[:, :, 0] = np.maximum(environment[:, :, 0], 0.0)

            pollinator_competition = 0.03 * self.intensity * dt_days * progress
            environment[:, :, 2] -= pollinator_competition
            environment[:, :, 2] = np.maximum(environment[:, :, 2], 0.0)

        elif self.species_type == "parasite":
            # Parasites weaken trees
            tree_weakening = 0.03 * self.intensity * dt_days * progress
            environment[:, :, 0] -= tree_weakening * environment[:, :, 0]  # Proportional damage
            environment[:, :, 0] = np.maximum(environment[:, :, 0], 0.0)

            # Affect carbon sequestration
            carbon_impact = 0.02 * self.intensity * dt_days * progress
            environment[:, :, 3] -= carbon_impact
            environment[:, :, 3] = np.maximum(environment[:, :, 3], 0.0)

        return True

    def get_description(self) -> str:
        return f"Invasive {self.species_type} (intensity: {self.intensity:.1f}, {self.elapsed_days:.1f}/{self.duration_days:.1f} days)"


class ClimateShiftEvent(DisturbanceEvent):
    """
    Climate shift disturbance - gradual changes in temperature and precipitation.
    """

    def __init__(self, intensity: float = 1.0, duration_days: float = 60.0, temp_change: float = 3.0):
        super().__init__(intensity, duration_days)
        self.temp_change = temp_change  # Degrees Celsius change

    def update(self, dt_days: float, environment: np.ndarray, weather: Dict[str, float]) -> bool:
        if not super().update(dt_days, environment, weather):
            return False

        progress = self.elapsed_days / self.duration_days

        # Gradual temperature increase
        temp_increase = self.temp_change * progress * dt_days / self.duration_days
        weather['temperature'] += temp_increase

        # Precipitation changes (often decreases in warming scenarios)
        precip_change = -0.5 * self.intensity * progress * dt_days / self.duration_days
        weather['precipitation'] = max(0, weather['precipitation'] + precip_change)

        # Effects on ecosystem
        temp_stress = abs(weather['temperature'] - 25) / 10  # Optimal ~25°C

        # Trees stressed by temperature
        tree_stress = 0.01 * temp_stress * self.intensity * dt_days
        environment[:, :, 0] -= tree_stress
        environment[:, :, 0] = np.maximum(environment[:, :, 0], 0.0)

        # Water availability affected
        water_stress = 0.02 * (1 - weather['precipitation']/5) * self.intensity * dt_days
        environment[:, :, 1] -= water_stress
        environment[:, :, 1] = np.maximum(environment[:, :, 1], 0.0)

        return True

    def get_description(self) -> str:
        return f"Climate shift (+{self.temp_change:.1f}°C, intensity: {self.intensity:.1f}, {self.elapsed_days:.1f}/{self.duration_days:.1f} days)"
