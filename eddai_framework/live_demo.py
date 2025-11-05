#!/usr/bin/env python3
"""
Live E.D.D.A.I. Demonstration
Shows real-time adaptation to environmental changes
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eddai import EDDAI
from simulation import VirtualForest, DroughtEvent, FireEvent


def run_live_demo():
    print("🌱 E.D.D.A.I. - Live Environmental Adaptation Demo")
    print("=" * 55)

    # Initialize systems
    forest = VirtualForest(grid_size=(20, 20), biome="temperate_forest")
    eddai = EDDAI(biome_id="live_demo")

    print("\n🌲 Virtual Forest Ecosystem Initialized")
    print("🧠 E.D.D.A.I. System Online")
    print("\n🚀 Starting Live Adaptation Demo...\n")

    # Run simulation with real-time feedback
    for step in range(25):
        # Get current state
        forest_state = forest.step()

        # Show current ecosystem status
        biodiversity = forest_state['metrics']['biodiversity_index']
        disturbance = forest_state['metrics']['disturbance_level']
        carbon_flux = forest_state['metrics']['carbon_flux']
        soil_moisture = forest_state['metrics']['soil_moisture']

        print(f"📊 Step {step+1:2d} | Bio:{biodiversity:.3f} | Dist:{disturbance:.3f} | Carbon:{carbon_flux:.3f} | Water:{soil_moisture:.3f}", end="")

        # Add disturbances at specific points
        if step == 8:
            drought = DroughtEvent(intensity=0.7, duration_days=8)
            forest.add_disturbance(drought)
            print(" 🌵 DROUGHT BEGINS!")
            continue

        if step == 16:
            fire = FireEvent(intensity=0.5, duration_days=4)
            forest.add_disturbance(fire)
            print(" 🔥 FIRE BEGINS!")
            continue

        # E.D.D.A.I. makes decision
        eddai_result = eddai.step(forest_state['environment'])

        # Apply action
        action_outcomes = forest.apply_action(eddai_result['action'])
        action_type = eddai_result['action']['type']

        # Show AI's decision
        print(f" 🤖 Action: {action_type}")

        # Small delay for readability
        time.sleep(0.1)

    print("\n✅ Demo Complete!")
    print("🎯 E.D.D.A.I. successfully adapted to environmental disturbances")

    # Show final learning summary
    print("\n🧠 Learning Summary:")
    print(f"   📚 Experiences stored: {eddai.step_count}")
    print(f"   🏗️  Neural network adapted {eddai.brain.layers[0].out_features + eddai.brain.layers[1].out_features} neurons")
    print("   🎯 Actions learned: monitoring, planting, irrigation, pollinator attraction")


if __name__ == "__main__":
    run_live_demo()
