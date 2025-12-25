"""
Experiment Manager - A/B testing framework.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Variant:
    """Experiment variant (A, B, etc.)."""
    name: str
    weight: int = 50  # Traffic percentage
    conversions: int = 0
    impressions: int = 0
    
    @property
    def conversion_rate(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions * 100


@dataclass
class Experiment:
    """A/B experiment definition."""
    id: str
    name: str
    description: str
    variants: List[Variant] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_sample_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "variants": [
                {
                    "name": v.name,
                    "weight": v.weight,
                    "conversions": v.conversions,
                    "impressions": v.impressions,
                    "conversion_rate": v.conversion_rate,
                }
                for v in self.variants
            ],
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


class ExperimentManager:
    """
    Manages A/B experiments with statistical analysis.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
    
    def create_experiment(
        self,
        id: str,
        name: str,
        description: str,
        variants: List[Dict[str, Any]] = None
    ) -> Experiment:
        """Create a new experiment."""
        if variants is None:
            variants = [
                {"name": "control", "weight": 50},
                {"name": "treatment", "weight": 50},
            ]
        
        variant_objs = [Variant(name=v["name"], weight=v["weight"]) for v in variants]
        
        experiment = Experiment(
            id=id,
            name=name,
            description=description,
            variants=variant_objs,
        )
        
        self.experiments[id] = experiment
        return experiment
    
    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.start_date = datetime.now()
        return exp
    
    def stop_experiment(self, experiment_id: str) -> Experiment:
        """Stop an experiment."""
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.end_date = datetime.now()
        return exp
    
    def get_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Get variant assignment for a user (deterministic)."""
        if experiment_id not in self.experiments:
            return None
        
        exp = self.experiments[experiment_id]
        if exp.status != ExperimentStatus.RUNNING:
            return None
        
        # Deterministic assignment based on hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        cumulative = 0
        for variant in exp.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant.name
        
        return exp.variants[-1].name if exp.variants else None
    
    def record_impression(self, experiment_id: str, variant_name: str) -> None:
        """Record an impression for a variant."""
        exp = self.experiments.get(experiment_id)
        if exp:
            for variant in exp.variants:
                if variant.name == variant_name:
                    variant.impressions += 1
                    break
    
    def record_conversion(self, experiment_id: str, variant_name: str) -> None:
        """Record a conversion for a variant."""
        exp = self.experiments.get(experiment_id)
        if exp:
            for variant in exp.variants:
                if variant.name == variant_name:
                    variant.conversions += 1
                    break
    
    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results with statistical analysis."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}
        
        results = exp.to_dict()
        
        # Calculate statistical significance for A/B
        if len(exp.variants) == 2:
            v1, v2 = exp.variants[0], exp.variants[1]
            results["statistical_significance"] = self._calculate_significance(v1, v2)
            results["winner"] = self._determine_winner(v1, v2)
        
        return results
    
    def _calculate_significance(self, v1: Variant, v2: Variant) -> float:
        """Calculate statistical significance using z-test."""
        if v1.impressions < 30 or v2.impressions < 30:
            return 0.0
        
        p1 = v1.conversions / v1.impressions if v1.impressions > 0 else 0
        p2 = v2.conversions / v2.impressions if v2.impressions > 0 else 0
        
        p_pool = (v1.conversions + v2.conversions) / (v1.impressions + v2.impressions)
        
        if p_pool == 0 or p_pool == 1:
            return 0.0
        
        se = math.sqrt(p_pool * (1 - p_pool) * (1/v1.impressions + 1/v2.impressions))
        
        if se == 0:
            return 0.0
        
        z = abs(p1 - p2) / se
        
        # Approximate p-value from z-score (simplified)
        if z > 2.576:
            return 99.0
        elif z > 1.96:
            return 95.0
        elif z > 1.645:
            return 90.0
        else:
            return z * 40  # Rough approximation
    
    def _determine_winner(self, v1: Variant, v2: Variant) -> Optional[str]:
        """Determine the winning variant."""
        if v1.impressions < 100 or v2.impressions < 100:
            return None
        
        significance = self._calculate_significance(v1, v2)
        if significance < 95:
            return None
        
        if v1.conversion_rate > v2.conversion_rate:
            return v1.name
        return v2.name
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        return [exp.to_dict() for exp in self.experiments.values()]
