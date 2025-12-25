"""
Load Testing Suite for Digital Twin Robotics Lab.

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000
"""

from .config import LOAD_SCENARIOS, SLA_DEFINITIONS, LoadProfile, LoadScenario, get_scenario

__all__ = ['LOAD_SCENARIOS', 'SLA_DEFINITIONS', 'LoadProfile', 'LoadScenario', 'get_scenario']
