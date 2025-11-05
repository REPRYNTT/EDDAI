#!/usr/bin/env python3
"""
E.D.D.A.I. Web Dashboard - Environmental AI Monitoring Platform
A Flask-based web application for environmental monitoring and AI insights
"""

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from datetime import datetime
import sys
import os

# Add the framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eddai import EDDAI
from simulation import VirtualForest

app = Flask(__name__)

# Global instances
forest = None
eddai_system = None

def initialize_systems():
    """Initialize the E.D.D.A.I. system and virtual forest"""
    global forest, eddai_system
    if forest is None:
        forest = VirtualForest(grid_size=(20, 20), biome="temperate_forest")
    if eddai_system is None:
        eddai_system = EDDAI(biome_id="web_dashboard")

@app.route('/')
def dashboard():
    """Main dashboard page"""
    initialize_systems()
    return render_template('dashboard.html')

@app.route('/api/ecosystem_status')
def get_ecosystem_status():
    """Get current ecosystem status"""
    initialize_systems()

    forest_state = forest.step()
    eddai_result = eddai_system.step(forest_state['environment'])

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'biodiversity_index': round(float(forest_state['metrics']['biodiversity_index']), 3),
            'carbon_flux': round(float(forest_state['metrics']['carbon_flux']), 3),
            'disturbance_level': round(float(forest_state['metrics']['disturbance_level']), 3),
            'soil_moisture': round(float(forest_state['metrics']['soil_moisture']), 3),
            'temperature': round(float(forest_state['metrics']['temperature']), 1),
            'humidity': round(float(forest_state['metrics']['humidity']), 1)
        },
        'ai_decision': {
            'action': eddai_result['action']['type'],
            'confidence': 0.85,  # Placeholder for now
            'reasoning': get_action_reasoning(eddai_result['action']['type'])
        },
        'sof_weights': {
            'human_yield': round(float(eddai_result['sof_weights']['human']), 3),
            'biodiversity': round(float(eddai_result['sof_weights']['fauna']), 3),
            'ecosystem_stability': round(float(eddai_result['sof_weights']['abiotic']), 3),
            'carbon_sequester': round(float(eddai_result['sof_weights']['flora']), 3)
        }
    })

@app.route('/api/simulate_disturbance', methods=['POST'])
def simulate_disturbance():
    """Simulate environmental disturbance"""
    data = request.get_json()
    disturbance_type = data.get('type', 'drought')
    intensity = data.get('intensity', 0.5)

    # This would integrate with the disturbance system
    # For now, return simulation results
    return jsonify({
        'success': True,
        'disturbance': disturbance_type,
        'intensity': intensity,
        'predicted_impact': {
            'biodiversity_change': -0.15 * intensity,
            'carbon_change': -0.2 * intensity,
            'recovery_time': f"{int(30 * intensity)} days"
        },
        'ai_recommendations': get_disturbance_recommendations(disturbance_type, intensity)
    })

@app.route('/api/get_recommendations')
def get_recommendations():
    """Get AI recommendations for current conditions"""
    initialize_systems()

    forest_state = forest.step()
    eddai_result = eddai_system.step(forest_state['environment'])

    recommendations = generate_recommendations(forest_state, eddai_result)

    return jsonify({
        'recommendations': recommendations,
        'priority_actions': ['monitor', 'plant', 'irrigate'][:3],  # Top 3 actions
        'environmental_health_score': calculate_health_score(forest_state['metrics'])
    })

def get_action_reasoning(action_type):
    """Get AI reasoning for specific actions"""
    reasonings = {
        'monitor': "Continuous assessment of ecosystem health and early disturbance detection",
        'plant': "Increasing vegetation cover to enhance biodiversity and carbon sequestration",
        'irrigate': "Maintaining soil moisture levels during dry periods",
        'attract_pollinators': "Supporting pollination networks and biodiversity",
        'sequester_carbon': "Optimizing carbon capture through vegetation management"
    }
    return reasonings.get(action_type, "Adaptive response to current environmental conditions")

def get_disturbance_recommendations(disturbance_type, intensity):
    """Get recommendations for specific disturbances"""
    recommendations = {
        'drought': [
            "Implement water conservation measures",
            "Plant drought-resistant species",
            "Monitor soil moisture levels closely",
            "Consider supplemental irrigation if critical"
        ],
        'fire': [
            "Create firebreaks and defensible spaces",
            "Plant fire-resistant vegetation",
            "Implement controlled burns if appropriate",
            "Monitor regeneration patterns"
        ],
        'flood': [
            "Improve drainage systems",
            "Plant flood-tolerant species",
            "Monitor erosion patterns",
            "Implement sediment control measures"
        ]
    }
    return recommendations.get(disturbance_type, ["Monitor situation closely", "Implement adaptive management strategies"])

def generate_recommendations(forest_state, eddai_result):
    """Generate comprehensive recommendations based on current state"""
    metrics = forest_state['metrics']

    recommendations = []

    # Biodiversity recommendations
    if metrics['biodiversity_index'] < 0.4:
        recommendations.append({
            'category': 'Biodiversity',
            'priority': 'High',
            'action': 'Increase habitat diversity',
            'impact': 'Enhance species richness and ecosystem resilience'
        })

    # Carbon recommendations
    if metrics['carbon_flux'] < 0.5:
        recommendations.append({
            'category': 'Carbon',
            'priority': 'Medium',
            'action': 'Expand carbon sequestration areas',
            'impact': 'Improve climate change mitigation'
        })

    # Water recommendations
    if metrics['soil_moisture'] < 0.8:
        recommendations.append({
            'category': 'Water',
            'priority': 'High',
            'action': 'Implement water conservation',
            'impact': 'Maintain ecosystem hydration levels'
        })

    return recommendations

def calculate_health_score(metrics):
    """Calculate overall environmental health score"""
    # Weighted combination of key metrics
    score = (
        metrics['biodiversity_index'] * 0.3 +
        (1 - metrics['disturbance_level']) * 0.3 +
        metrics['carbon_flux'] * 0.2 +
        metrics['soil_moisture'] * 0.2
    )
    return round(score * 100, 1)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
