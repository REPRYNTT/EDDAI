#!/usr/bin/env python3
"""
E.D.D.A.I. Disturbance Response Demonstration

This script shows how E.D.D.A.I. adapts to various environmental disturbances
like droughts, fires, and floods.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eddai import EDDAI
from simulation import VirtualForest
from simulation.disturbance_events import DroughtEvent, FireEvent, FloodEvent, InvasiveSpeciesEvent


def run_disturbance_scenario(forest, eddai, disturbance, scenario_name, steps=30):
    """Run a single disturbance scenario and collect data."""
    print(f"\n🌀 Running {scenario_name} scenario...")

    metrics_history = []
    actions_history = []
    disturbance_active = False

    for step in range(steps):
        # Get current forest state
        forest_state = forest.step()

        # Add disturbance at step 5
        if step == 5 and not disturbance_active:
            forest.add_disturbance(disturbance)
            disturbance_active = True
            print(f"  📍 Introduced disturbance at step {step}")

        # E.D.D.A.I. responds
        eddai_result = eddai.step(forest_state['environment'])
        action_outcomes = forest.apply_action(eddai_result['action'])

        # Record data
        metrics_history.append(forest_state['metrics'].copy())
        actions_history.append({
            'type': eddai_result['action']['type'],
            'intensity': eddai_result['action']['intensity'],
            'step': step
        })

        # Progress indicator
        if (step + 1) % 10 == 0:
            print(f"  ✅ Step {step + 1}/{steps} completed")

    return metrics_history, actions_history


def main():
    print("🌪️  E.D.D.A.I. Disturbance Response Demonstration")
    print("=" * 60)

    # Initialize systems
    forest = VirtualForest(grid_size=(20, 20), biome="temperate_forest")
    eddai = EDDAI(biome_id="temperate_forest")

    # Define disturbance scenarios
    scenarios = [
        (DroughtEvent(intensity=0.8, duration_days=10), "Drought Response"),
        (FireEvent(intensity=0.7, duration_days=5), "Fire Response"),
        (FloodEvent(intensity=0.6, duration_days=7), "Flood Response"),
        (InvasiveSpeciesEvent(intensity=0.5, duration_days=15, species_type="herbivore"), "Invasive Species Response")
    ]

    # Run all scenarios
    all_results = {}
    for disturbance, scenario_name in scenarios:
        # Reset forest and E.D.D.A.I. for each scenario
        forest = VirtualForest(grid_size=(20, 20), biome="temperate_forest")
        eddai = EDDAI(biome_id="temperate_forest")

        metrics, actions = run_disturbance_scenario(forest, eddai, disturbance, scenario_name)
        all_results[scenario_name] = {
            'metrics': metrics,
            'actions': actions
        }

    # Create comparative analysis
    print("\n📊 Comparative Analysis:")
    print("-" * 40)

    for scenario_name, results in all_results.items():
        metrics = results['metrics']
        actions = results['actions']

        # Calculate key metrics
        initial_biodiversity = metrics[0]['biodiversity_index']
        final_biodiversity = metrics[-1]['biodiversity_index']
        min_biodiversity = min(m['biodiversity_index'] for m in metrics)
        recovery_rate = (final_biodiversity - min_biodiversity) / (initial_biodiversity - min_biodiversity) if initial_biodiversity != min_biodiversity else 1.0

        # Count actions
        action_counts = {}
        for action in actions:
            action_type = action['type']
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        most_common_action = max(action_counts.items(), key=lambda x: x[1])

        print(f"\n{scenario_name}:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(f"  Most common action: {most_common_action[0]} ({most_common_action[1]} times)")

    # Generate visualization
    print("\n📈 Generating comparative plots...")
    create_comparison_plots(all_results)
    print("💾 Saved comparison to 'disturbance_response_comparison.png'")


def create_comparison_plots(all_results):
    """Create comparative plots for all disturbance scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('E.D.D.A.I. Disturbance Response Comparison', fontsize=16)

    scenario_names = list(all_results.keys())
    colors = ['blue', 'red', 'green', 'orange']

    # Plot 1: Biodiversity trajectories
    ax = axes[0, 0]
    for i, (scenario, results) in enumerate(all_results.items()):
        metrics = results['metrics']
        steps = range(len(metrics))
        biodiversity = [m['biodiversity_index'] for m in metrics]
        ax.plot(steps, biodiversity, label=scenario, color=colors[i], linewidth=2)

        # Mark disturbance onset
        ax.axvline(x=5, color='black', linestyle='--', alpha=0.5, label='Disturbance onset' if i == 0 else "")

    ax.set_title('Biodiversity Index Over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Biodiversity Index')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Disturbance level trajectories
    ax = axes[0, 1]
    for i, (scenario, results) in enumerate(all_results.items()):
        metrics = results['metrics']
        steps = range(len(metrics))
        disturbance = [m['disturbance_level'] for m in metrics]
        ax.plot(steps, disturbance, label=scenario, color=colors[i], linewidth=2)

        ax.axvline(x=5, color='black', linestyle='--', alpha=0.5)

    ax.set_title('Disturbance Level Over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Disturbance Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Action distribution
    ax = axes[1, 0]
    action_types = ['monitor', 'irrigate', 'plant', 'attract_pollinators', 'harvest']
    x = np.arange(len(scenario_names))
    width = 0.15

    for i, action_type in enumerate(action_types):
        counts = []
        for scenario in scenario_names:
            actions = all_results[scenario]['actions']
            count = sum(1 for a in actions if a['type'] == action_type)
            counts.append(count)

        ax.bar(x + i * width, counts, width, label=action_type, alpha=0.7)

    ax.set_title('Action Distribution by Scenario')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Action Count')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Recovery analysis
    ax = axes[1, 1]
    recovery_data = []
    scenario_labels = []

    for scenario, results in all_results.items():
        metrics = results['metrics']
        biodiversity = [m['biodiversity_index'] for m in metrics]

        # Calculate recovery metrics
        pre_disturbance = np.mean(biodiversity[:5])  # Steps 0-4
        post_disturbance = np.mean(biodiversity[-5:])  # Last 5 steps
        min_during = min(biodiversity[5:])  # After disturbance onset

        recovery = (post_disturbance - min_during) / (pre_disturbance - min_during) if pre_disturbance != min_during else 1.0
        impact = (min_during - pre_disturbance) / pre_disturbance if pre_disturbance > 0 else 0

        recovery_data.append([recovery, impact])
        scenario_labels.append(scenario)

    recovery_data = np.array(recovery_data)

    # Plot recovery vs impact
    scatter = ax.scatter(recovery_data[:, 1], recovery_data[:, 0],
                        s=100, c=colors[:len(scenario_labels)], alpha=0.7)

    for i, label in enumerate(scenario_labels):
        ax.annotate(label, (recovery_data[i, 1], recovery_data[i, 0]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title('Recovery vs Impact Analysis')
    ax.set_xlabel('Relative Impact (negative = more severe)')
    ax.set_ylabel('Recovery Rate')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('disturbance_response_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
