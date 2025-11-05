"""
Epigenetic Place Memory - Long-term spatiotemporal embeddings that encode disturbance history.
"""

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict


class EpigeneticPlaceMemory:
    """
    AI systems retain contextual embeddings of the specific biomes they inhabit,
    akin to ecological memory in ecosystems post-disturbance.

    Stores historical disturbance regimes and uses transformer-based world models
    for geospatial attention and pattern recognition.
    """

    def __init__(self, biome_id: str, memory_capacity: int = 1000, embedding_dim: int = 64):
        """
        Initialize epigenetic place memory.

        Args:
            biome_id: Unique identifier for the biome
            memory_capacity: Maximum number of memories to store
            embedding_dim: Dimension of embeddings
        """
        self.biome_id = biome_id
        self.memory_capacity = memory_capacity
        self.embedding_dim = embedding_dim

        # Memory storage: timestamp -> (state_embedding, outcome_embedding, attention_weights)
        self.memories: Dict[datetime, Dict[str, np.ndarray]] = {}
        self.memory_order: List[datetime] = []

        # Embedding networks
        self.state_encoder = nn.Sequential(
            nn.Linear(10, 128),  # Input: ecological state vector
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        self.outcome_encoder = nn.Sequential(
            nn.Linear(5, 64),    # Input: outcome vector
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Attention mechanism for geospatial patterns
        self.attention_net = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)

        # Disturbance pattern recognition
        self.disturbance_patterns = defaultdict(list)  # Pattern type -> list of embeddings

        # Memory consolidation parameters
        self.consolidation_threshold = 0.7
        self.forgetting_rate = 0.01

    def encode(self, timestamp: datetime, ecological_state: Dict[str, float], outcome: Dict[str, float]):
        """
        Encode and store a new memory.

        Args:
            timestamp: When the event occurred
            ecological_state: Ecological conditions at the time
            outcome: Results of actions taken
        """
        # Convert to tensors and encode
        state_vector = torch.tensor([
            ecological_state.get('biodiversity_index', 0),
            ecological_state.get('carbon_flux', 0),
            ecological_state.get('soil_respiration', 0),
            ecological_state.get('phenological_shift', 0),
            ecological_state.get('disturbance_level', 0),
            ecological_state.get('temperature', 25),
            ecological_state.get('humidity', 60),
            ecological_state.get('soil_moisture', 0.5),
            ecological_state.get('wind_speed', 5),
            ecological_state.get('light_intensity', 0.8)
        ], dtype=torch.float32)

        outcome_vector = torch.tensor([
            outcome.get('human_yield', 0),
            outcome.get('flora_health', 0),
            outcome.get('fauna_health', 0),
            outcome.get('abiotic_stability', 0),
            outcome.get('carbon_sequestered', 0)
        ], dtype=torch.float32)

        with torch.no_grad():
            state_embedding = self.state_encoder(state_vector).numpy()
            outcome_embedding = self.outcome_encoder(outcome_vector).numpy()

        # Store memory
        memory_entry = {
            'state_embedding': state_embedding,
            'outcome_embedding': outcome_embedding,
            'ecological_state': ecological_state.copy(),
            'outcome': outcome.copy(),
            'attention_weights': self._calculate_attention_weights(ecological_state)
        }

        self.memories[timestamp] = memory_entry
        self.memory_order.append(timestamp)

        # Maintain capacity
        if len(self.memories) > self.memory_capacity:
            oldest = self.memory_order.pop(0)
            del self.memories[oldest]

        # Classify and store disturbance patterns
        self._classify_disturbance(timestamp, ecological_state, outcome)

        # Consolidate memories periodically
        if len(self.memories) % 50 == 0:
            self._consolidate_memories()

    def retrieve(self, query_state: Dict[str, float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on current ecological state.

        Args:
            query_state: Current ecological conditions
            k: Number of memories to retrieve

        Returns:
            List of relevant memory entries with similarity scores
        """
        if not self.memories:
            return []

        # Encode query
        query_vector = torch.tensor([
            query_state.get('biodiversity_index', 0),
            query_state.get('carbon_flux', 0),
            query_state.get('soil_respiration', 0),
            query_state.get('phenological_shift', 0),
            query_state.get('disturbance_level', 0),
            query_state.get('temperature', 25),
            query_state.get('humidity', 60),
            query_state.get('soil_moisture', 0.5),
            query_state.get('wind_speed', 5),
            query_state.get('light_intensity', 0.8)
        ], dtype=torch.float32)

        with torch.no_grad():
            query_embedding = self.state_encoder(query_vector).numpy()

        # Calculate similarities
        similarities = []
        for timestamp, memory in self.memories.items():
            similarity = np.dot(query_embedding, memory['state_embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory['state_embedding'])
            )
            similarities.append((timestamp, similarity, memory))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                'timestamp': ts,
                'similarity': sim,
                'memory': mem,
                'age_days': (datetime.now() - ts).days
            }
            for ts, sim, mem in similarities[:k]
        ]

    def get_disturbance_context(self, disturbance_type: str) -> Optional[Dict[str, Any]]:
        """
        Get learned context for specific disturbance types.

        Args:
            disturbance_type: Type of disturbance (e.g., 'drought', 'fire', 'flood')

        Returns:
            Aggregated pattern information or None if no pattern found
        """
        if disturbance_type not in self.disturbance_patterns:
            return None

        patterns = self.disturbance_patterns[disturbance_type]
        if not patterns:
            return None

        # Aggregate patterns
        state_embeddings = np.array([p['state_embedding'] for p in patterns])
        outcome_embeddings = np.array([p['outcome_embedding'] for p in patterns])

        return {
            'count': len(patterns),
            'avg_state_embedding': np.mean(state_embeddings, axis=0),
            'avg_outcome_embedding': np.mean(outcome_embeddings, axis=0),
            'state_variance': np.var(state_embeddings, axis=0),
            'typical_response': self._extract_typical_response(patterns)
        }

    def _calculate_attention_weights(self, ecological_state: Dict[str, float]) -> np.ndarray:
        """Calculate attention weights for different ecological aspects."""
        # Weight based on disturbance level and biodiversity
        disturbance = ecological_state.get('disturbance_level', 0)
        biodiversity = ecological_state.get('biodiversity_index', 0.5)

        weights = np.array([
            biodiversity * 0.8 + (1 - disturbance) * 0.2,  # Biodiversity focus
            disturbance * 0.9,                             # Disturbance focus
            biodiversity * 0.6 + disturbance * 0.4,        # Phenology focus
            0.7,                                           # Carbon focus
            disturbance * 0.5                              # Soil focus
        ])

        return weights / weights.sum()  # Normalize

    def _classify_disturbance(self, timestamp: datetime, ecological_state: Dict[str, float], outcome: Dict[str, float]):
        """Classify the type of disturbance or ecological event."""
        disturbance_level = ecological_state.get('disturbance_level', 0)
        biodiversity = ecological_state.get('biodiversity_index', 0.5)
        soil_moisture = ecological_state.get('soil_moisture', 0.5)

        # Simple classification rules
        if disturbance_level > 0.7:
            if soil_moisture < 0.3:
                disturbance_type = 'drought'
            elif biodiversity < 0.3:
                disturbance_type = 'fire'
            else:
                disturbance_type = 'flood'
        elif biodiversity < 0.4:
            disturbance_type = 'habitat_loss'
        elif disturbance_level > 0.3:
            disturbance_type = 'stress_event'
        else:
            disturbance_type = 'normal'

        # Store pattern
        self.disturbance_patterns[disturbance_type].append({
            'timestamp': timestamp,
            'state_embedding': self.memories[timestamp]['state_embedding'],
            'outcome_embedding': self.memories[timestamp]['outcome_embedding'],
            'ecological_state': ecological_state,
            'outcome': outcome
        })

    def _consolidate_memories(self):
        """Consolidate similar memories to prevent redundancy."""
        if len(self.memories) < 10:
            return

        # Simple consolidation: remove very similar recent memories
        recent_timestamps = self.memory_order[-20:]  # Last 20 memories

        to_remove = []
        for i, ts1 in enumerate(recent_timestamps):
            for ts2 in recent_timestamps[i+1:]:
                if ts2 in to_remove:
                    continue

                emb1 = self.memories[ts1]['state_embedding']
                emb2 = self.memories[ts2]['state_embedding']

                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if similarity > self.consolidation_threshold:
                    # Keep the more recent one
                    if ts1 < ts2:
                        to_remove.append(ts1)
                    else:
                        to_remove.append(ts2)
                    break

        # Remove consolidated memories
        for ts in to_remove:
            if ts in self.memories:
                del self.memories[ts]
                self.memory_order.remove(ts)

    def _extract_typical_response(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract typical response pattern from disturbance history."""
        outcomes = [p['outcome'] for p in patterns]

        return {
            'avg_yield_impact': np.mean([o.get('human_yield', 0) for o in outcomes]),
            'avg_biodiversity_recovery': np.mean([o.get('fauna_health', 0) for o in outcomes]),
            'avg_carbon_change': np.mean([o.get('carbon_sequestered', 0) for o in outcomes]),
            'recovery_time_days': np.mean([p.get('age_days', 30) for p in patterns])
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            'total_memories': len(self.memories),
            'disturbance_types': list(self.disturbance_patterns.keys()),
            'pattern_counts': {k: len(v) for k, v in self.disturbance_patterns.items()},
            'oldest_memory': min(self.memory_order) if self.memory_order else None,
            'newest_memory': max(self.memory_order) if self.memory_order else None,
            'memory_capacity': self.memory_capacity
        }
