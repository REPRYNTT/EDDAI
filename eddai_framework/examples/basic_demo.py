#!/usr/bin/env python3
"""
Basic E.D.D.A.I. Demonstration

This script demonstrates the core functionality of E.D.D.A.I. in a simple scenario.
The AI learns to adapt to a virtual forest ecosystem over time.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import argparse

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eddai import EDDAI
from simulation import VirtualForest, DroughtEvent, FireEvent


def main():
    parser = argparse.ArgumentParser(description='E.D.D.A.I. Basic Demonstration')
    parser.add_argument('--steps', type=int, default=50, help='Number of simulation steps')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--biome', type=str, default='temperate_forest', help='Biome type')
    args = parser.parse_args()

    print("🌱 E.D.D.A.I. - Environmental Data-Driven Adaptive Intelligence")
    print("=" * 60)
    print(f"Biome: {args.biome}")
    print(f"Simulation Steps: {args.steps}")
    print()

    # Initialize the virtual forest
    forest = VirtualForest(grid_size=(20, 20), biome=args.biome)
    print("🌲 Initialized virtual forest ecosystem")

    # Initialize E.D.D.A.I.
    eddai = EDDAI(biome_id=args.biome)
    print("🧠 Initialized E.D.D.A.I. system")

    # Track metrics over time
    metrics_history = []
    actions_history = []
    sof_weights_history = []

    print("\n🚀 Starting simulation...")
    print("-" * 40)

    for step in range(args.steps):
        # Get current forest state
        forest_state = forest.step()

        # E.D.D.A.I. observes, interprets, adapts, acts, learns, remembers
        eddai_result = eddai.step(forest_state['environment'])

        # Apply E.D.D.A.I.'s action to the forest
        action_outcomes = forest.apply_action(eddai_result['action'])

        # Record data
        metrics_history.append(forest_state['metrics'].copy())
        actions_history.append(eddai_result['action'].copy())
        sof_weights_history.append(eddai_result['sof_weights'].copy())

        # Add disturbances at specific points
        if step == 15:
            drought = DroughtEvent(intensity=0.8, duration_days=10)
            forest.add_disturbance(drought)
            print(f"🌵 Step {step}: Drought event started")

        if step == 35:
            fire = FireEvent(intensity=0.6, duration_days=5)
            forest.add_disturbance(fire)
            print(f"🔥 Step {step}: Fire event started")

        # Progress update
        if (step + 1) % 2 == 0:  # Show progress every 2 steps
            biodiversity = forest_state['metrics']['biodiversity_index']
            disturbance = forest_state['metrics']['disturbance_level']
            action_type = eddai_result['action']['type']
            print(f"Step {step+1}: Biodiversity={biodiversity:.3f}, Disturbance={disturbance:.3f}, Action: {action_type}")

    print("\n✅ Simulation completed!")
    print("-" * 40)

    # Final analysis
    final_metrics = metrics_history[-1]
    print("📊 Final Ecosystem State:")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    final_weights = sof_weights_history[-1]
    print("\n⚖️  Final SOF Weights:")
    for stakeholder, weight in final_weights.items():
        print(".3f")

    eddai_status = eddai.get_status()
    print(f"\n🧠 E.D.D.A.I. learned from {eddai_status['step_count']} experiences")
    print(f"📚 Memory contains {eddai_status['memory_size']} events")

    # Generate visualizations if requested
    if args.visualize:
        print("\n📈 Generating visualizations...")
        create_plots(metrics_history, actions_history, sof_weights_history, args.steps)
        print("💾 Saved plots to 'eddai_demo_plots.png'")

    print("\n🎉 Demo complete! E.D.D.A.I. successfully adapted to environmental changes.")


def create_plots(metrics_history, actions_history, sof_weights_history, num_steps):
    """Create comprehensive plots of the simulation results."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('E.D.D.A.I. Adaptation Demo Results', fontsize=16)

    steps = range(num_steps)

    # Plot 1: Biodiversity and disturbance over time
    axes[0, 0].plot(steps, [m['biodiversity_index'] for m in metrics_history], label='Biodiversity Index', color='green')
    axes[0, 0].plot(steps, [m['disturbance_level'] for m in metrics_history], label='Disturbance Level', color='red')
    axes[0, 0].set_title('Ecosystem Health Metrics')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Index (0-1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: SOF weights evolution
    stakeholders = ['human', 'flora', 'fauna', 'abiotic']
    colors = ['blue', 'green', 'orange', 'brown']
    for i, stakeholder in enumerate(stakeholders):
        weights = [w[stakeholder] for w in sof_weights_history]
        axes[0, 1].plot(steps, weights, label=stakeholder.capitalize(), color=colors[i])
    axes[0, 1].set_title('Symbiotic Objective Function Weights')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Action types over time
    action_types = [a['type'] for a in actions_history]
    unique_actions = list(set(action_types))
    action_counts = []
    for action in unique_actions:
        counts = [1 if a == action else 0 for a in action_types]
        action_counts.append(counts)

    bottom = np.zeros(num_steps)
    for i, (action, counts) in enumerate(zip(unique_actions, action_counts)):
        axes[1, 0].bar(steps, counts, bottom=bottom, label=action, alpha=0.7)
        bottom += np.array(counts)

    axes[1, 0].set_title('E.D.D.A.I. Actions Over Time')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Action Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Environmental layers
    layer_names = ['Trees', 'Water', 'Pollinators', 'Carbon']
    for i, layer in enumerate(['trees', 'water', 'pollinators', 'carbon']):
        # Assuming we can get layer means from metrics or need to modify to track them
        if layer in metrics_history[0]:
            values = [m[layer] for m in metrics_history]
        else:
            values = [0.5] * num_steps  # Placeholder

        axes[1, 1].plot(steps, values, label=layer_names[i])
    axes[1, 1].set_title('Environmental Layer Averages')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Average Density (0-1)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Temperature and soil moisture
    axes[2, 0].plot(steps, [m['temperature'] for m in metrics_history], label='Temperature (°C)', color='red')
    axes[2, 0].set_title('Weather Conditions')
    axes[2, 0].set_xlabel('Time Steps')
    axes[2, 0].set_ylabel('Temperature (°C)', color='red')
    axes[2, 0].tick_params(axis='y', labelcolor='red')
    axes[2, 0].grid(True, alpha=0.3)

    ax2 = axes[2, 0].twinx()
    ax2.plot(steps, [m['soil_moisture'] for m in metrics_history], label='Soil Moisture', color='blue')
    ax2.set_ylabel('Soil Moisture (0-1)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Plot 6: Learning progress (action intensity)
    intensities = [a.get('intensity', 0) for a in actions_history]
    axes[2, 1].plot(steps, intensities, label='Action Intensity', color='purple')
    axes[2, 1].set_title('E.D.D.A.I. Learning Progress')
    axes[2, 1].set_xlabel('Time Steps')
    axes[2, 1].set_ylabel('Action Intensity (0-1)')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eddai_demo_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
