"""
Virtual Forest - Dynamic ecosystem simulation for E.D.D.A.I. testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import random


class VirtualForest:
    """
    Dynamic forest ecosystem simulation with trees, water, pollinators, and carbon cycles.

    Provides a realistic environment for testing E.D.D.A.I. adaptation capabilities.
    """

    def __init__(self, grid_size: Tuple[int, int] = (20, 20), biome: str = "temperate_forest"):
        """
        Initialize the virtual forest.

        Args:
            grid_size: Size of the forest grid (rows, cols)
            biome: Type of biome to simulate
        """
        self.grid_size = grid_size
        self.biome = biome
        self.current_time = datetime.now()

        # Environmental layers: [trees, water, pollinators, carbon]
        self.environment = np.zeros((*grid_size, 4), dtype=np.float32)

        # Initialize with realistic starting conditions
        self._initialize_ecosystem()

        # Environmental parameters
        self.weather_params = {
            'temperature': 20.0,  # Celsius
            'humidity': 0.6,      # 0-1
            'wind_speed': 2.0,    # m/s
            'precipitation': 0.0, # mm/day
            'light_intensity': 0.8  # 0-1
        }

        # Ecosystem dynamics parameters
        self.dynamics_params = {
            'tree_growth_rate': 0.02,
            'tree_decay_rate': 0.01,
            'water_evaporation': 0.05,
            'water_infiltration': 0.1,
            'pollinator_migration': 0.03,
            'carbon_sequestration': 0.015,
            'carbon_emission': 0.008
        }

        # Seasonal parameters
        self.seasonal_cycle = 0  # 0-365 (days)
        self.seasonal_amplitude = {
            'temperature': 15.0,
            'precipitation': 5.0,
            'light': 0.3
        }

        # Disturbance tracking
        self.active_disturbances = []
        self.disturbance_history = []

    def _initialize_ecosystem(self):
        """Initialize the forest with realistic starting conditions."""
        rows, cols = self.grid_size

        # Trees: Higher in center, vary randomly
        center_row, center_col = rows // 2, cols // 2
        for i in range(rows):
            for j in range(cols):
                distance_from_center = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                base_density = max(0, 0.8 - distance_from_center * 0.03)
                self.environment[i, j, 0] = np.clip(np.random.normal(base_density, 0.2), 0, 1)

        # Water: More in lower areas, some variation
        for i in range(rows):
            for j in range(cols):
                base_water = 0.3 + (rows - i) * 0.02  # More water at bottom
                self.environment[i, j, 1] = np.clip(np.random.normal(base_water, 0.15), 0, 1)

        # Pollinators: Follow tree density with some independence
        for i in range(rows):
            for j in range(cols):
                tree_influence = self.environment[i, j, 0] * 0.7
                random_factor = np.random.normal(0.2, 0.1)
                self.environment[i, j, 2] = np.clip(tree_influence + random_factor, 0, 1)

        # Carbon: Related to tree biomass
        for i in range(rows):
            for j in range(cols):
                self.environment[i, j, 3] = self.environment[i, j, 0] * 0.6 + np.random.normal(0.2, 0.1)
                self.environment[i, j, 3] = np.clip(self.environment[i, j, 3], 0, 1)

    def step(self, time_delta: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """
        Advance the simulation by one time step.

        Args:
            time_delta: How much time to advance

        Returns:
            Dict containing environmental state and metrics
        """
        self.current_time += time_delta
        hours_passed = time_delta.total_seconds() / 3600

        # Update seasonal cycle
        self.seasonal_cycle = (self.seasonal_cycle + hours_passed / 24) % 365

        # Update weather
        self._update_weather()

        # Apply natural dynamics
        self._apply_ecosystem_dynamics(hours_passed)

        # Apply active disturbances
        self._apply_disturbances(hours_passed)

        # Calculate ecological metrics
        metrics = self._calculate_ecological_metrics()

        return {
            'timestamp': self.current_time,
            'environment': self.environment.copy(),
            'weather': self.weather_params.copy(),
            'metrics': metrics,
            'seasonal_cycle': self.seasonal_cycle,
            'active_disturbances': len(self.active_disturbances)
        }

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply an E.D.D.A.I. action to the environment.

        Args:
            action: Action dict from EDDAI

        Returns:
            Dict with action outcomes
        """
        action_type = action.get('type', 'monitor')
        intensity = action.get('intensity', 0.1)

        outcomes = {
            'human_yield': 0.0,
            'flora_health': 0.0,
            'fauna_health': 0.0,
            'abiotic_stability': 0.0,
            'carbon_sequestered': 0.0
        }

        if action_type == 'irrigate':
            # Add water to dry areas
            dry_mask = self.environment[:, :, 1] < 0.4
            water_added = intensity * 0.2 * dry_mask
            self.environment[:, :, 1] += water_added
            self.environment[:, :, 1] = np.clip(self.environment[:, :, 1], 0, 1)
            outcomes['abiotic_stability'] = np.mean(water_added) * 5

        elif action_type == 'plant':
            # Plant trees in sparse areas
            sparse_mask = self.environment[:, :, 0] < 0.5
            trees_added = intensity * 0.15 * sparse_mask
            self.environment[:, :, 0] += trees_added
            self.environment[:, :, 0] = np.clip(self.environment[:, :, 0], 0, 1)
            outcomes['flora_health'] = np.mean(trees_added) * 4
            outcomes['carbon_sequestered'] = np.mean(trees_added) * 3

        elif action_type == 'attract_pollinators':
            # Increase pollinator presence
            low_poll_mask = self.environment[:, :, 2] < 0.6
            pollinators_added = intensity * 0.1 * low_poll_mask
            self.environment[:, :, 2] += pollinators_added
            self.environment[:, :, 2] = np.clip(self.environment[:, :, 2], 0, 1)
            outcomes['fauna_health'] = np.mean(pollinators_added) * 6

        elif action_type == 'harvest':
            # Human harvest - increases yield but may decrease other metrics
            harvestable = self.environment[:, :, 0] > 0.6
            harvest_amount = intensity * 0.1 * harvestable
            self.environment[:, :, 0] -= harvest_amount
            outcomes['human_yield'] = np.mean(harvest_amount) * 8
            outcomes['flora_health'] = -np.mean(harvest_amount) * 2  # Negative impact
            outcomes['carbon_sequestered'] = -np.mean(harvest_amount) * 1.5

        # Ensure bounds
        self.environment = np.clip(self.environment, 0, 1)

        return outcomes

    def add_disturbance(self, disturbance):
        """Add a disturbance event to the simulation."""
        self.active_disturbances.append(disturbance)
        self.disturbance_history.append({
            'type': disturbance.__class__.__name__,
            'start_time': self.current_time,
            'intensity': getattr(disturbance, 'intensity', 1.0)
        })

    def _update_weather(self):
        """Update weather conditions based on seasonal cycle."""
        day_of_year = self.seasonal_cycle

        # Temperature: sinusoidal with seasonal variation
        base_temp = 15 + self.seasonal_amplitude['temperature'] * np.sin(2 * np.pi * day_of_year / 365)
        self.weather_params['temperature'] = base_temp + np.random.normal(0, 2)

        # Precipitation: higher in certain seasons
        precip_base = 2 + self.seasonal_amplitude['precipitation'] * max(0, np.sin(2 * np.pi * (day_of_year - 45) / 365))
        self.weather_params['precipitation'] = max(0, precip_base + np.random.normal(0, 1))

        # Humidity: correlated with precipitation
        self.weather_params['humidity'] = min(1.0, 0.4 + self.weather_params['precipitation'] * 0.1 + np.random.normal(0, 0.1))

        # Light intensity: higher in summer
        light_base = 0.5 + self.seasonal_amplitude['light'] * max(0, np.sin(2 * np.pi * (day_of_year - 80) / 365))
        self.weather_params['light_intensity'] = light_base + np.random.normal(0, 0.1)

        # Wind speed: random with occasional gusts
        self.weather_params['wind_speed'] = max(0, 2 + np.random.normal(0, 1.5))

    def _apply_ecosystem_dynamics(self, hours: float):
        """Apply natural ecosystem dynamics."""
        dt = hours / 24  # Convert to days

        # Tree dynamics
        light_factor = self.weather_params['light_intensity']
        water_factor = np.mean(self.environment[:, :, 1])
        temp_factor = max(0, 1 - abs(self.weather_params['temperature'] - 25) / 20)

        tree_growth = self.dynamics_params['tree_growth_rate'] * light_factor * water_factor * temp_factor * dt
        tree_decay = self.dynamics_params['tree_decay_rate'] * (1 - water_factor) * dt

        self.environment[:, :, 0] += tree_growth - tree_decay
        self.environment[:, :, 0] = np.clip(self.environment[:, :, 0], 0, 1)

        # Water dynamics
        precipitation = self.weather_params['precipitation'] * dt
        evaporation = self.dynamics_params['water_evaporation'] * (1 - self.weather_params['humidity']) * dt

        self.environment[:, :, 1] += precipitation - evaporation
        self.environment[:, :, 1] = np.clip(self.environment[:, :, 1], 0, 1)

        # Pollinator dynamics
        tree_density = np.mean(self.environment[:, :, 0])
        temp_suitability = max(0, 1 - abs(self.weather_params['temperature'] - 22) / 15)

        pollinator_change = self.dynamics_params['pollinator_migration'] * tree_density * temp_suitability * dt
        self.environment[:, :, 2] += pollinator_change
        self.environment[:, :, 2] = np.clip(self.environment[:, :, 2], 0, 1)

        # Carbon dynamics
        sequestration = self.dynamics_params['carbon_sequestration'] * tree_density * dt
        emission = self.dynamics_params['carbon_emission'] * (1 - water_factor) * dt

        self.environment[:, :, 3] += sequestration - emission
        self.environment[:, :, 3] = np.clip(self.environment[:, :, 3], 0, 1)

    def _apply_disturbances(self, hours: float):
        """Apply active disturbance events."""
        dt = hours / 24

        # Process each disturbance
        remaining_disturbances = []
        for disturbance in self.active_disturbances:
            if disturbance.update(dt, self.environment, self.weather_params):
                remaining_disturbances.append(disturbance)

        self.active_disturbances = remaining_disturbances

    def _calculate_ecological_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive ecological metrics."""
        trees = self.environment[:, :, 0]
        water = self.environment[:, :, 1]
        pollinators = self.environment[:, :, 2]
        carbon = self.environment[:, :, 3]

        return {
            'biodiversity_index': (np.mean(pollinators) * 0.5 + np.mean(trees) * 0.3 + np.std(trees) * 0.2),
            'carbon_flux': np.mean(carbon),
            'soil_respiration': np.mean(water) * 0.7 + np.mean(trees) * 0.3,
            'phenological_shift': np.mean(trees) * 0.8 + np.mean(pollinators) * 0.2,
            'disturbance_level': 1.0 - (np.mean(trees) + np.mean(water) + np.mean(pollinators) + np.mean(carbon)) / 4.0,
            'temperature': self.weather_params['temperature'],
            'humidity': self.weather_params['humidity'],
            'soil_moisture': np.mean(water),
            'wind_speed': self.weather_params['wind_speed'],
            'light_intensity': self.weather_params['light_intensity']
        }

    def visualize(self, save_path: Optional[str] = None):
        """Create a visualization of the current forest state."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Virtual Forest - {self.current_time.strftime("%Y-%m-%d %H:%M")}\nBiome: {self.biome}')

        layer_names = ['Trees', 'Water', 'Pollinators', 'Carbon']
        for i, (ax, layer_name) in enumerate(zip(axes.flat, layer_names)):
            im = ax.imshow(self.environment[:, :, i], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'{layer_name} Density')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current forest state."""
        metrics = self._calculate_ecological_metrics()

        return {
            'timestamp': self.current_time,
            'biome': self.biome,
            'grid_size': self.grid_size,
            'seasonal_cycle': self.seasonal_cycle,
            'weather': self.weather_params,
            'metrics': metrics,
            'active_disturbances': len(self.active_disturbances),
            'layer_means': {
                'trees': float(np.mean(self.environment[:, :, 0])),
                'water': float(np.mean(self.environment[:, :, 1])),
                'pollinators': float(np.mean(self.environment[:, :, 2])),
                'carbon': float(np.mean(self.environment[:, :, 3]))
            }
        }
