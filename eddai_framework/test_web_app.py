#!/usr/bin/env python3
"""
Quick test to verify the web app loads correctly
"""

import sys
import os
import time
import threading

# Add the framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

def test_web_app():
    """Test that the web app initializes correctly"""

    # Test that the app can be created
    assert app is not None, "Flask app should be created"

    # Test that routes are registered
    with app.test_client() as client:
        # Test main route
        response = client.get('/')
        assert response.status_code == 200, "Dashboard route should work"

        # Test API routes
        response = client.get('/api/ecosystem_status')
        assert response.status_code == 200, "API route should work"

        # Test disturbance simulation
        response = client.post('/api/simulate_disturbance',
                              json={'type': 'drought', 'intensity': 0.5})
        assert response.status_code == 200, "Disturbance simulation should work"

    print("✅ Web app test passed! All routes working correctly.")

if __name__ == "__main__":
    test_web_app()
