"""
Plastic Neural Fabric - Self-rewiring neural architectures that adapt to environmental gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


class PlasticNeuralFabric(nn.Module):
    """
    Neural architecture that grows, prunes, or rewires based on ecological state changes.

    Implements topological plasticity - the network physically deforms in response to
    environmental gradients, mimicking natural adaptation patterns.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], output_dim: int = 10):
        """
        Initialize the plastic neural fabric.

        Args:
            input_dim: Dimension of input features
            hidden_dims: Initial hidden layer dimensions
            output_dim: Dimension of output actions
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Dynamic layer management
        self.layers = nn.ModuleList()
        self.layer_masks = []  # For pruning/growth
        self.plasticity_threshold = 0.1  # Threshold for topology changes

        # Initialize with base architecture
        self._build_initial_network(hidden_dims)

        # Plasticity parameters
        self.growth_rate = 0.05
        self.pruning_rate = 0.03
        self.modulation_lr = 0.01

        # Attention for different modalities
        self.modality_attention = nn.Parameter(torch.ones(4))  # trees, water, pollinators, carbon

    def _build_initial_network(self, hidden_dims: list):
        """Build the initial neural network."""
        prev_dim = self.input_dim

        for dim in hidden_dims:
            layer = nn.Linear(prev_dim, dim)
            self.layers.append(layer)
            self.layer_masks.append(torch.ones(dim))  # Initially all neurons active
            prev_dim = dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        self.output_mask = torch.ones(self.output_dim)

    def adapt_topology(self, ecological_state: Dict[str, float]):
        """
        Adapt network topology based on ecological state.

        Args:
            ecological_state: Current ecological conditions
        """
        # Calculate environmental gradients
        disturbance_level = ecological_state.get('disturbance_level', 0.0)
        biodiversity_index = ecological_state.get('biodiversity_index', 0.5)
        phenological_shift = ecological_state.get('phenological_shift', 0.5)

        # Modulate attention based on current needs
        with torch.no_grad():
            # Increase attention to water during droughts (high disturbance, low biodiversity)
            water_attention = disturbance_level * 0.5 + (1 - biodiversity_index) * 0.3
            # Increase attention to pollinators during phenological shifts
            pollinator_attention = phenological_shift * 0.4 + biodiversity_index * 0.3
            # Increase attention to trees during low biodiversity
            tree_attention = (1 - biodiversity_index) * 0.5
            # Carbon attention based on overall health
            carbon_attention = biodiversity_index * 0.6

            new_attention = torch.tensor([tree_attention, water_attention, pollinator_attention, carbon_attention])
            self.modality_attention.copy_(torch.clamp(self.modality_attention + self.modulation_lr * (new_attention - self.modality_attention), 0.1, 2.0))

        # Prune neurons in low-attention areas
        if disturbance_level > self.plasticity_threshold:
            self._prune_neurons(disturbance_level)

        # Grow neurons in high-biodiversity areas
        if biodiversity_index > 0.7:
            self._grow_neurons(biodiversity_index)

    def _prune_neurons(self, intensity: float):
        """Prune neurons based on current attention patterns."""
        prune_prob = self.pruning_rate * intensity

        for i, mask in enumerate(self.layer_masks):
            # Prune based on attention to different modalities
            modality_weights = torch.abs(self.layers[i].weight).mean(dim=1)  # Average weight per neuron
            prune_mask = torch.rand_like(mask) > prune_prob * (1 - modality_weights / modality_weights.max())
            self.layer_masks[i] = mask * prune_mask.float()

    def _grow_neurons(self, intensity: float):
        """Grow new neurons in underutilized areas."""
        growth_prob = self.growth_rate * intensity

        for i, (layer, mask) in enumerate(zip(self.layers, self.layer_masks)):
            if torch.rand(1) < growth_prob:
                # Add a new neuron
                new_neurons = 1
                current_out = layer.out_features

                # Expand weight matrices
                new_weight = torch.randn(new_neurons, layer.in_features) * 0.1
                layer.weight = nn.Parameter(torch.cat([layer.weight, new_weight], dim=0))
                layer.out_features += new_neurons  # Update the layer's output dimension

                if layer.bias is not None:
                    new_bias = torch.zeros(new_neurons)
                    layer.bias = nn.Parameter(torch.cat([layer.bias, new_bias], dim=0))

                # Update mask
                new_mask = torch.ones(new_neurons)
                self.layer_masks[i] = torch.cat([mask, new_mask], dim=0)

                # Update next layer if exists
                if i < len(self.layers) - 1:
                    next_layer = self.layers[i + 1]
                    new_next_weight = torch.randn(next_layer.out_features, new_neurons) * 0.1
                    next_layer.weight = nn.Parameter(torch.cat([next_layer.weight, new_next_weight], dim=1))
                    next_layer.in_features += new_neurons  # Update the next layer's input dimension
                else:
                    # If this is the last hidden layer, update the output layer
                    new_output_weight = torch.randn(self.output_dim, new_neurons) * 0.1
                    self.output_layer.weight = nn.Parameter(torch.cat([self.output_layer.weight, new_output_weight], dim=1))
                    self.output_layer.in_features += new_neurons

    def forward(self, env_data: Dict[str, np.ndarray],
                memory: Optional[Any] = None,
                sof: Optional[Any] = None) -> Dict[str, Any]:
        """
        Forward pass with adaptive computation.

        Args:
            env_data: Environmental sensor data
            memory: Epigenetic place memory (for context)
            sof: Symbiotic objective function (for guidance)

        Returns:
            Dict containing action recommendations
        """
        # Extract and modulate input features
        trees = torch.tensor(env_data['trees'], dtype=torch.float32).flatten()
        water = torch.tensor(env_data['water'], dtype=torch.float32).flatten()
        pollinators = torch.tensor(env_data['pollinators'], dtype=torch.float32).flatten()
        carbon = torch.tensor(env_data['carbon'], dtype=torch.float32).flatten()

        # Apply modality attention
        modulated_features = torch.stack([
            trees * self.modality_attention[0],
            water * self.modality_attention[1],
            pollinators * self.modality_attention[2],
            carbon * self.modality_attention[3]
        ])

        x = modulated_features.flatten()

        # Forward through plastic layers
        for layer, mask in zip(self.layers, self.layer_masks):
            x = layer(x)
            x = x * mask  # Apply pruning mask
            x = F.relu(x)

        # Output layer
        action_logits = self.output_layer(x)
        action_logits = action_logits * self.output_mask

        # Convert to action recommendations
        action_probs = F.softmax(action_logits, dim=0)

        # Map to specific actions (simplified)
        actions = self._logits_to_actions(action_logits)

        return {
            'action_logits': action_logits.detach().numpy(),
            'action_probs': action_probs.detach().numpy(),
            'type': actions['type'],
            'intensity': actions['intensity'],
            'modality_attention': self.modality_attention.detach().numpy()
        }

    def _logits_to_actions(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Convert network outputs to interpretable actions."""
        # Simplified action mapping
        action_idx = torch.argmax(logits).item()

        action_types = ['monitor', 'irrigate', 'plant', 'attract_pollinators', 'sequester_carbon']
        if action_idx >= len(action_types):
            action_type = 'monitor'
        else:
            action_type = action_types[action_idx]

        # Intensity based on logit magnitude
        intensity = torch.sigmoid(logits[action_idx]).item()

        return {
            'type': action_type,
            'intensity': intensity
        }

    def get_topology_info(self) -> Dict[str, Any]:
        """Get information about current network topology."""
        return {
            'num_layers': len(self.layers),
            'layer_sizes': [mask.sum().item() for mask in self.layer_masks],
            'total_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'modality_attention': self.modality_attention.detach().numpy().tolist()
        }
