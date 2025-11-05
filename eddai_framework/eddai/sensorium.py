"""
Environmental Sensorium - Multimodal sensory data ingestion system.
"""

import numpy as np
from typing import Dict, Any


class EnvironmentalSensorium:
    """
    Multimodal sensory mesh that continuously ingests high-dimensional ecological data.

    In real deployment, this would interface with:
    - IoT soil probes (Decagon Teros)
    - Bioacoustic recorders (AudioMoth)
    - Hyperspectral cameras (drones)
    - eDNA samplers
    - Weather stations
    """

    def __init__(self, grid_size: tuple = (20, 20)):
        """
        Initialize the sensorium.

        Args:
            grid_size: Size of the environmental grid (rows, cols)
        """
        self.grid_size = grid_size
        self.sensors = {
            'hyperspectral': {'active': True, 'noise': 0.05},
            'bioacoustic': {'active': True, 'noise': 0.1},
            'edaphic': {'active': True, 'noise': 0.08},
            'atmospheric': {'active': True, 'noise': 0.03},
            'phenocam': {'active': True, 'noise': 0.07}
        }

    def read(self, environment_state: np.ndarray) -> Dict[str, Any]:
        """
        Read current environmental state through multiple sensor modalities.

        Args:
            environment_state: Current state of the environment grid
                           Shape: (rows, cols, 4) for [trees, water, pollinators, carbon]

        Returns:
            Dict containing processed sensor data
        """
        if environment_state.shape[:2] != self.grid_size:
            raise ValueError(f"Environment state shape {environment_state.shape[:2]} doesn't match grid_size {self.grid_size}")

        # Extract layers
        trees = environment_state[:, :, 0]
        water = environment_state[:, :, 1]
        pollinators = environment_state[:, :, 2]
        carbon = environment_state[:, :, 3]

        # Simulate multimodal sensing
        sensor_data = {}

        # Hyperspectral: Vegetation indices, water content
        if self.sensors['hyperspectral']['active']:
            sensor_data['hyperspectral'] = {
                'ndvi': self._calculate_ndvi(trees, water),
                'water_index': self._add_noise(water * 0.8 + trees * 0.2, self.sensors['hyperspectral']['noise']),
                'chlorophyll_content': self._add_noise(trees * 0.9, self.sensors['hyperspectral']['noise'])
            }

        # Bioacoustic: Species vocalizations, insect activity
        if self.sensors['bioacoustic']['active']:
            sensor_data['bioacoustic'] = {
                'bird_vocalization_complexity': self._add_noise(pollinators * 0.7 + trees * 0.3, self.sensors['bioacoustic']['noise']),
                'insect_buzz_index': self._add_noise(pollinators * 0.8, self.sensors['bioacoustic']['noise']),
                'amphibian_calls': self._add_noise(water * 0.6 + trees * 0.4, self.sensors['bioacoustic']['noise'])
            }

        # Edaphic: Soil moisture, nutrients, microbial activity
        if self.sensors['edaphic']['active']:
            sensor_data['edaphic'] = {
                'soil_moisture': self._add_noise(water * 0.85 + trees * 0.15, self.sensors['edaphic']['noise']),
                'microbial_respiration': self._add_noise(trees * 0.5 + water * 0.3 + carbon * 0.2, self.sensors['edaphic']['noise']),
                'nutrient_availability': self._add_noise((trees + water) * 0.5, self.sensors['edaphic']['noise'])
            }

        # Atmospheric: CO2, temperature, humidity
        if self.sensors['atmospheric']['active']:
            sensor_data['atmospheric'] = {
                'co2_concentration': self._add_noise(carbon * 0.9 + 0.1, self.sensors['atmospheric']['noise']),
                'temperature': self._add_noise(25 + trees * 2 - water * 1, self.sensors['atmospheric']['noise']),
                'humidity': self._add_noise(water * 0.7 + 40, self.sensors['atmospheric']['noise'])
            }

        # Phenocam: Leaf area, phenological stages
        if self.sensors['phenocam']['active']:
            sensor_data['phenocam'] = {
                'leaf_area_index': self._add_noise(trees * 0.95, self.sensors['phenocam']['noise']),
                'phenological_stage': self._add_noise(trees * 0.8 + pollinators * 0.2, self.sensors['phenocam']['noise']),
                'canopy_cover': self._add_noise(trees * 0.9, self.sensors['phenocam']['noise'])
            }

        # Aggregate into simplified layers for downstream processing
        sensor_data['trees'] = self._add_noise(trees, 0.05)
        sensor_data['water'] = self._add_noise(water, 0.03)
        sensor_data['pollinators'] = self._add_noise(pollinators, 0.08)
        sensor_data['carbon'] = self._add_noise(carbon, 0.04)

        return sensor_data

    def _calculate_ndvi(self, trees: np.ndarray, water: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index."""
        nir = trees * 0.8 + 0.2  # Simulated NIR reflectance
        red = trees * 0.6 + water * 0.1 + 0.3  # Simulated red reflectance
        return (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero

    def _add_noise(self, data: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic sensor noise."""
        noise = np.random.normal(0, noise_level, data.shape)
        return np.clip(data + noise, 0, 1)

    def calibrate_sensors(self, calibration_data: Dict[str, Any]):
        """
        Calibrate sensors using known reference data.

        Args:
            calibration_data: Reference measurements for calibration
        """
        # In real implementation, this would adjust sensor parameters
        # For simulation, we'll just update noise levels based on calibration
        for sensor_type, data in calibration_data.items():
            if sensor_type in self.sensors:
                # Reduce noise based on calibration quality
                self.sensors[sensor_type]['noise'] *= 0.8

    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors."""
        return {
            sensor: info.copy()
            for sensor, info in self.sensors.items()
        }
