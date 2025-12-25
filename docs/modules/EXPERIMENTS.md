# A/B Testing Module

## Overview
Experimentation framework for controlled feature testing.

## Files

### `experiment.py`
Manages A/B experiments with statistical analysis.

```python
class ExperimentManager:
    def create_experiment(id, name, description, variants) -> Experiment
    def start_experiment(experiment_id) -> Experiment
    def stop_experiment(experiment_id) -> Experiment
    def get_variant(experiment_id, user_id) -> str
    def record_impression(experiment_id, variant_name)
    def record_conversion(experiment_id, variant_name)
    def get_results(experiment_id) -> Dict
```

**Experiment Statuses:**
- DRAFT, RUNNING, PAUSED, COMPLETED

**Statistical Analysis:**
- Z-test for significance
- Confidence levels: 90%, 95%, 99%
- Automatic winner determination

## Usage

```python
from experiments import ExperimentManager

manager = ExperimentManager()

# Create A/B test
manager.create_experiment(
    id="nav-button-color",
    name="Navigation Button Color Test",
    description="Test blue vs green buttons",
    variants=[
        {"name": "blue", "weight": 50},
        {"name": "green", "weight": 50},
    ]
)

manager.start_experiment("nav-button-color")

# Assign user to variant
variant = manager.get_variant("nav-button-color", user_id="user123")
# Returns "blue" or "green" (deterministic per user)

# Track metrics
manager.record_impression("nav-button-color", variant)
if user_clicked:
    manager.record_conversion("nav-button-color", variant)

# Get results
results = manager.get_results("nav-button-color")
# Includes: conversion rates, statistical significance, winner
```

## Features
- Deterministic user assignment (same user = same variant)
- Weighted traffic splitting
- Built-in statistical significance calculation
- Automatic winner detection at 95% confidence
