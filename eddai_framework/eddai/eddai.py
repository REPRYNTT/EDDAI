"""
Main EDDAI class - Orchestrates the adaptive intelligence system.
"""

import numpy as np
from datetime import datetime
from .sensorium import EnvironmentalSensorium
from .neural_fabric import PlasticNeuralFabric
from .place_memory import EpigeneticPlaceMemory
from .sof import SymbioticObjectiveFunction


class EDDAI:
    """
    Environmental Data-Driven Adaptive Intelligence

    A self-evolving AI system that embeds ecological dynamics into its core learning loop,
    enabling real-time adaptation to planetary patterns and cultivation of symbiotic intelligence
    with natural systems.
    """

    def __init__(self, biome_id: str, grid_size: tuple = (20, 20)):
        """
        Initialize the EDDAI system.

        Args:
            biome_id: Unique identifier for the biome (e.g., "amazon_rainforest")
            grid_size: Size of the environmental grid (rows, cols)
        """
        self.biome_id = biome_id
        self.timestamp = datetime.now()

        # Core components
        self.sensorium = EnvironmentalSensorium(grid_size)
        self.brain = PlasticNeuralFabric(input_dim=grid_size[0] * grid_size[1] * 4)  # 4 features: trees, water, pollinators, carbon
        self.memory = EpigeneticPlaceMemory(biome_id)
        self.sof = SymbioticObjectiveFunction()

        # State tracking
        self.step_count = 0

    def step(self, environment_state: np.ndarray) -> dict:
        """
        Execute one adaptive cycle: Observe → Interpret → Adapt → Act → Learn → Remember

        Args:
            environment_state: Current state of the environment grid

        Returns:
            Dict containing actions taken and outcomes
        """
        self.step_count += 1
        current_time = datetime.now()

        # 1. Observe: Ingest environmental data
        env_data = self.sensorium.read(environment_state)

        # 2. Interpret: Map to ecological state
        ecological_state = self._interpret_state(env_data)

        # 3. Adapt: Modify architecture, objectives, and behavior
        self.brain.adapt_topology(ecological_state)
        self.sof.adapt_weights(ecological_state)

        # 4. Act: Generate and execute actions
        action = self.brain.forward(env_data, self.memory, self.sof)
        outcome = self._execute_action(action, environment_state)

        # 5. Learn: Update SOF from multi-species outcomes
        self.sof.update(outcome)

        # 6. Remember: Store disturbance regime in epigenetic memory
        self.memory.encode(current_time, ecological_state, outcome)

        return {
            "timestamp": current_time,
            "ecological_state": ecological_state,
            "action": action,
            "outcome": outcome,
            "sof_weights": self.sof.weights.copy()
        }

    def _interpret_state(self, env_data: dict) -> dict:
        """Interpret raw sensor data into ecological state vectors."""
        # Simplified interpretation - in real implementation, this would use ML
        trees = np.mean(env_data['trees'])
        water = np.mean(env_data['water'])
        pollinators = np.mean(env_data['pollinators'])
        carbon = np.mean(env_data['carbon'])

        return {
            "biodiversity_index": pollinators * 0.5 + trees * 0.3 + water * 0.2,
            "carbon_flux": carbon,
            "soil_respiration": water * 0.7 + trees * 0.3,
            "phenological_shift": trees * 0.8 + pollinators * 0.2,
            "disturbance_level": 1.0 - (trees + water + pollinators + carbon) / 4.0
        }

    def _execute_action(self, action: dict, environment_state: np.ndarray) -> dict:
        """
        Execute actions in the environment (simplified for simulation).

        In real deployment, this would interface with drones, valves, etc.
        """
        # Simplified action execution
        action_type = action.get('type', 'monitor')
        intensity = action.get('intensity', 0.1)

        if action_type == 'irrigate':
            # Increase water in dry areas
            dry_mask = environment_state[:, :, 1] < 0.3  # water layer
            environment_state[dry_mask, 1] += intensity * 0.2

        elif action_type == 'plant':
            # Add trees in sparse areas
            sparse_mask = environment_state[:, :, 0] < 0.4  # tree layer
            environment_state[sparse_mask, 0] += intensity * 0.15

        elif action_type == 'attract_pollinators':
            # Increase pollinator presence
            low_poll_mask = environment_state[:, :, 2] < 0.5
            environment_state[low_poll_mask, 2] += intensity * 0.1

        # Calculate outcomes
        new_biodiversity = np.mean(environment_state[:, :, 2])  # pollinators proxy
        new_carbon = np.mean(environment_state[:, :, 3])
        new_yield = np.mean(environment_state[:, :, 0])  # trees proxy

        return {
            "human_yield": new_yield,
            "flora_health": np.mean(environment_state[:, :, 0]),
            "fauna_health": new_biodiversity,
            "abiotic_stability": np.mean(environment_state[:, :, 1]),
            "carbon_sequestered": new_carbon
        }

    def get_status(self) -> dict:
        """Get current system status."""
        return {
            "biome_id": self.biome_id,
            "step_count": self.step_count,
            "sof_weights": self.sof.weights,
            "brain_topology": self.brain.get_topology_info(),
            "memory_size": len(self.memory.memories)
        }
