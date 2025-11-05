"""
Symbiotic Objective Function - Dynamic utility model balancing human, biotic, and abiotic goals.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class SymbioticObjectiveFunction:
    """
    Dynamic utility function that balances human, biotic, and abiotic goals.

    Formalization: U = w₁·U_human + w₂·U_flora + w₃·U_fauna + w₄·U_abiotic

    Weights are dynamically negotiated via reinforcement learning from environmental feedback loops.
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the symbiotic objective function.

        Args:
            initial_weights: Initial weights for different stakeholders
        """
        if initial_weights is None:
            initial_weights = {
                'human': 0.4,      # Economic yield, human benefits
                'flora': 0.3,      # Plant health, biodiversity
                'fauna': 0.2,      # Animal welfare, pollinators
                'abiotic': 0.1     # Soil, water, climate stability
            }

        self.weights = initial_weights.copy()
        self._normalize_weights()

        # Reinforcement learning components
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.reward_history = []

        # Adaptive components
        self.weight_adapter = nn.Sequential(
            nn.Linear(10, 32),  # Input: ecological state + outcomes
            nn.ReLU(),
            nn.Linear(32, 4),   # Output: weight adjustments for 4 stakeholders
            nn.Softmax(dim=0)
        )

        # Constraints and thresholds
        self.constraints = {
            'min_biodiversity_threshold': 0.4,  # Minimum acceptable biodiversity
            'max_carbon_emission': 0.3,         # Maximum carbon emission tolerance
            'water_sustainability': 0.5,        # Minimum water sustainability
            'soil_health_minimum': 0.6          # Minimum soil health
        }

        # Ethical guardrails
        self.ethical_vetoes = {
            'biodiversity_crash': False,
            'habitat_destruction': False,
            'carbon_overload': False
        }

    def __call__(self, outcomes: Dict[str, float], ecological_state: Dict[str, float]) -> float:
        """
        Calculate the overall symbiotic utility.

        Args:
            outcomes: Results of actions (yield, health metrics, etc.)
            ecological_state: Current ecological conditions

        Returns:
            Utility score between 0 and 1
        """
        # Check ethical vetoes first
        if self._check_ethical_vetoes(ecological_state, outcomes):
            return 0.0  # Complete veto - unacceptable action

        # Calculate stakeholder utilities
        u_human = self._calculate_human_utility(outcomes)
        u_flora = self._calculate_flora_utility(outcomes, ecological_state)
        u_fauna = self._calculate_fauna_utility(outcomes, ecological_state)
        u_abiotic = self._calculate_abiotic_utility(outcomes, ecological_state)

        # Apply weights
        utility = (
            self.weights['human'] * u_human +
            self.weights['flora'] * u_flora +
            self.weights['fauna'] * u_fauna +
            self.weights['abiotic'] * u_abiotic
        )

        # Apply constraints
        utility = self._apply_constraints(utility, ecological_state, outcomes)

        return np.clip(utility, 0.0, 1.0)

    def adapt_weights(self, ecological_state: Dict[str, float]):
        """
        Adapt weights based on ecological feedback using reinforcement learning.

        Args:
            ecological_state: Current ecological conditions
        """
        # Prepare input vector
        state_vector = torch.tensor([
            ecological_state.get('biodiversity_index', 0.5),
            ecological_state.get('carbon_flux', 0),
            ecological_state.get('soil_respiration', 0.5),
            ecological_state.get('phenological_shift', 0.5),
            ecological_state.get('disturbance_level', 0),
            ecological_state.get('temperature', 25),
            ecological_state.get('humidity', 60),
            ecological_state.get('soil_moisture', 0.5),
            ecological_state.get('wind_speed', 5),
            ecological_state.get('light_intensity', 0.8)
        ], dtype=torch.float32)

        # Get weight adjustments
        with torch.no_grad():
            adjustments = self.weight_adapter(state_vector)
            adjustments = adjustments.numpy()

        # Apply adjustments with momentum
        for i, stakeholder in enumerate(['human', 'flora', 'fauna', 'abiotic']):
            # Increase weight if ecological state supports it
            adjustment = (adjustments[i] - 0.25) * self.learning_rate  # Center around 0
            self.weights[stakeholder] += adjustment

        # Ensure weights stay positive and normalize
        self._normalize_weights()

        # Apply ecological pressure - reduce human weight during high disturbance
        disturbance = ecological_state.get('disturbance_level', 0)
        if disturbance > 0.6:
            # Shift weight from human to ecological stakeholders
            human_reduction = disturbance * 0.1
            self.weights['human'] = max(0.1, self.weights['human'] - human_reduction)

            # Redistribute to others
            redistribution = human_reduction / 3
            for stakeholder in ['flora', 'fauna', 'abiotic']:
                self.weights[stakeholder] += redistribution

        self._normalize_weights()

    def update(self, outcomes: Dict[str, float]):
        """
        Update the SOF based on action outcomes using reinforcement learning.

        Args:
            outcomes: Results of the actions taken
        """
        # Calculate reward based on symbiotic balance
        reward = self._calculate_reward(outcomes)

        self.reward_history.append(reward)

        # Update learning rate based on performance
        if len(self.reward_history) > 10:
            recent_avg = np.mean(self.reward_history[-10:])
            if recent_avg > 0.7:
                self.learning_rate *= 0.95  # Reduce learning rate when performing well
            elif recent_avg < 0.4:
                self.learning_rate *= 1.05  # Increase learning rate when struggling

        # Store reward for future weight adaptation
        self.last_reward = reward

    def _calculate_human_utility(self, outcomes: Dict[str, float]) -> float:
        """Calculate utility for human stakeholders."""
        yield_utility = outcomes.get('human_yield', 0)
        # Humans also benefit from stable ecosystems (indirect benefits)
        stability_bonus = outcomes.get('abiotic_stability', 0) * 0.2

        return np.clip(yield_utility + stability_bonus, 0, 1)

    def _calculate_flora_utility(self, outcomes: Dict[str, float], ecological_state: Dict[str, float]) -> float:
        """Calculate utility for plant life and vegetation."""
        flora_health = outcomes.get('flora_health', 0)
        biodiversity = ecological_state.get('biodiversity_index', 0.5)
        soil_health = ecological_state.get('soil_respiration', 0.5)

        # Plants benefit from good soil and biodiversity
        return np.clip((flora_health * 0.6 + biodiversity * 0.3 + soil_health * 0.1), 0, 1)

    def _calculate_fauna_utility(self, outcomes: Dict[str, float], ecological_state: Dict[str, float]) -> float:
        """Calculate utility for animal life and pollinators."""
        fauna_health = outcomes.get('fauna_health', 0)
        pollinator_index = ecological_state.get('biodiversity_index', 0.5) * 0.8  # Proxy

        # Animals benefit from biodiversity and food sources
        return np.clip((fauna_health * 0.7 + pollinator_index * 0.3), 0, 1)

    def _calculate_abiotic_utility(self, outcomes: Dict[str, float], ecological_state: Dict[str, float]) -> float:
        """Calculate utility for abiotic factors (soil, water, climate)."""
        abiotic_stability = outcomes.get('abiotic_stability', 0)
        carbon_sequestered = outcomes.get('carbon_sequestered', 0)
        water_sustainability = ecological_state.get('soil_moisture', 0.5)

        # Abiotic systems benefit from stability and carbon sequestration
        return np.clip((abiotic_stability * 0.5 + carbon_sequestered * 0.3 + water_sustainability * 0.2), 0, 1)

    def _check_ethical_vetoes(self, ecological_state: Dict[str, float], outcomes: Dict[str, float]) -> bool:
        """Check if current state violates ethical guardrails."""
        biodiversity = ecological_state.get('biodiversity_index', 0.5)
        carbon = outcomes.get('carbon_sequestered', 0)
        abiotic = outcomes.get('abiotic_stability', 0)

        # Biodiversity crash veto
        if biodiversity < self.constraints['min_biodiversity_threshold'] * 0.5:
            self.ethical_vetoes['biodiversity_crash'] = True
            return True

        # Carbon overload veto
        if carbon < -self.constraints['max_carbon_emission']:
            self.ethical_vetoes['carbon_overload'] = True
            return True

        # Habitat destruction veto (inferred from low flora/fauna health)
        flora_health = outcomes.get('flora_health', 0)
        fauna_health = outcomes.get('fauna_health', 0)
        if flora_health < 0.3 and fauna_health < 0.3:
            self.ethical_vetoes['habitat_destruction'] = True
            return True

        # Clear vetoes if conditions improve
        if biodiversity > self.constraints['min_biodiversity_threshold'] * 0.8:
            self.ethical_vetoes['biodiversity_crash'] = False
        if carbon > -self.constraints['max_carbon_emission'] * 0.5:
            self.ethical_vetoes['carbon_overload'] = False
        if flora_health > 0.5 and fauna_health > 0.5:
            self.ethical_vetoes['habitat_destruction'] = False

        return False

    def _apply_constraints(self, utility: float, ecological_state: Dict[str, float], outcomes: Dict[str, float]) -> float:
        """Apply hard constraints to the utility function."""
        # Biodiversity constraint
        biodiversity = ecological_state.get('biodiversity_index', 0.5)
        if biodiversity < self.constraints['min_biodiversity_threshold']:
            penalty = (self.constraints['min_biodiversity_threshold'] - biodiversity) * 2
            utility -= penalty

        # Carbon constraint
        carbon = outcomes.get('carbon_sequestered', 0)
        if carbon < -self.constraints['max_carbon_emission']:
            penalty = (-self.constraints['max_carbon_emission'] - carbon) * 1.5
            utility -= penalty

        # Water sustainability
        water = outcomes.get('abiotic_stability', 0)
        if water < self.constraints['water_sustainability']:
            penalty = (self.constraints['water_sustainability'] - water) * 1.2
            utility -= penalty

        return max(0, utility)

    def _calculate_reward(self, outcomes: Dict[str, float]) -> float:
        """Calculate reinforcement learning reward."""
        # Reward based on balanced outcomes
        total_outcome = sum(outcomes.values())
        outcome_balance = 1 - np.var(list(outcomes.values()))  # Lower variance = more balanced

        return np.clip((total_outcome * 0.7 + outcome_balance * 0.3), 0, 1)

    def _normalize_weights(self):
        """Ensure weights sum to 1 and are non-negative."""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] = max(0, self.weights[key] / total)
        else:
            # Reset to equal weights if all became zero
            equal_weight = 1.0 / len(self.weights)
            for key in self.weights:
                self.weights[key] = equal_weight

    def get_sof_status(self) -> Dict[str, Any]:
        """Get current status of the symbiotic objective function."""
        return {
            'weights': self.weights.copy(),
            'constraints': self.constraints.copy(),
            'ethical_vetoes': self.ethical_vetoes.copy(),
            'learning_rate': self.learning_rate,
            'recent_reward_avg': np.mean(self.reward_history[-10:]) if self.reward_history else 0,
            'total_rewards': len(self.reward_history)
        }
