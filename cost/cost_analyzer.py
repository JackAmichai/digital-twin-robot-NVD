"""
Cost Analyzer - Calculate and optimize cloud costs.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


@dataclass
class ResourceCost:
    """Cost for a specific resource."""
    resource_type: str
    resource_id: str
    hourly_cost: float
    monthly_cost: float
    utilization_percent: float
    
    @property
    def is_underutilized(self) -> bool:
        return self.utilization_percent < 30


# Pricing (simplified, USD per hour)
PRICING = {
    CloudProvider.AWS: {
        "gpu_instance": 3.06,      # p3.2xlarge
        "cpu_instance": 0.17,      # t3.xlarge
        "storage_gb": 0.023 / 730, # EBS per hour
    },
    CloudProvider.GCP: {
        "gpu_instance": 2.48,      # n1-standard-8 + T4
        "cpu_instance": 0.15,
        "storage_gb": 0.020 / 730,
    },
    CloudProvider.AZURE: {
        "gpu_instance": 2.07,      # NC6s_v3
        "cpu_instance": 0.17,
        "storage_gb": 0.024 / 730,
    },
}


class CostAnalyzer:
    """
    Analyze and optimize cloud infrastructure costs.
    """
    
    def __init__(self, provider: CloudProvider = CloudProvider.AWS):
        self.provider = provider
        self.pricing = PRICING[provider]
    
    def calculate_costs(self, resources: List[Dict[str, Any]]) -> List[ResourceCost]:
        """Calculate costs for resources."""
        costs = []
        
        for r in resources:
            res_type = r.get("type", "cpu_instance")
            hourly = self.pricing.get(res_type, 0.10)
            
            cost = ResourceCost(
                resource_type=res_type,
                resource_id=r.get("id", "unknown"),
                hourly_cost=hourly,
                monthly_cost=hourly * 730,
                utilization_percent=r.get("utilization", 50),
            )
            costs.append(cost)
        
        return costs
    
    def get_optimization_recommendations(
        self,
        costs: List[ResourceCost]
    ) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations."""
        recommendations = []
        
        for cost in costs:
            if cost.is_underutilized:
                recommendations.append({
                    "resource_id": cost.resource_id,
                    "issue": "underutilized",
                    "current_utilization": cost.utilization_percent,
                    "recommendation": "Consider downsizing or using spot instances",
                    "potential_savings": cost.monthly_cost * 0.4,
                })
        
        # Check for reserved instance opportunities
        total_monthly = sum(c.monthly_cost for c in costs)
        if total_monthly > 500:
            recommendations.append({
                "type": "reserved_instances",
                "recommendation": "Consider 1-year reserved instances for 30% savings",
                "potential_savings": total_monthly * 0.30,
            })
        
        return recommendations
    
    def estimate_gpu_costs(
        self,
        gpu_hours_per_day: float,
        days: int = 30
    ) -> Dict[str, float]:
        """Estimate GPU compute costs."""
        hourly = self.pricing["gpu_instance"]
        total_hours = gpu_hours_per_day * days
        
        return {
            "on_demand_cost": hourly * total_hours,
            "spot_cost": hourly * total_hours * 0.3,  # ~70% savings
            "reserved_cost": hourly * total_hours * 0.6,  # ~40% savings
            "total_hours": total_hours,
        }
    
    def generate_report(self, costs: List[ResourceCost]) -> Dict[str, Any]:
        """Generate cost report."""
        total_monthly = sum(c.monthly_cost for c in costs)
        recommendations = self.get_optimization_recommendations(costs)
        potential_savings = sum(r.get("potential_savings", 0) for r in recommendations)
        
        return {
            "report_date": datetime.now().isoformat(),
            "provider": self.provider.value,
            "total_monthly_cost": total_monthly,
            "resource_count": len(costs),
            "underutilized_count": sum(1 for c in costs if c.is_underutilized),
            "recommendations": recommendations,
            "potential_monthly_savings": potential_savings,
        }
