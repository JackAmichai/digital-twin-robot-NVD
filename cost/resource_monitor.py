"""
Resource Monitor - Track CPU, GPU, memory usage.
"""

import subprocess
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    disk_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "disk_percent": self.disk_percent,
        }


class ResourceMonitor:
    """
    Monitor system resource usage.
    """
    
    def __init__(self):
        self.history: List[ResourceUsage] = []
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        import psutil
        
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        gpu_percent = None
        gpu_mem = None
        
        try:
            gpu_info = self._get_gpu_usage()
            if gpu_info:
                gpu_percent = gpu_info.get("utilization")
                gpu_mem = gpu_info.get("memory_used_gb")
        except Exception:
            pass
        
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_used_gb=mem.used / (1024**3),
            gpu_percent=gpu_percent,
            gpu_memory_used_gb=gpu_mem,
            disk_percent=disk.percent,
        )
        
        self.history.append(usage)
        return usage
    
    def _get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """Get NVIDIA GPU usage via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "utilization": float(parts[0]),
                    "memory_used_gb": float(parts[1]) / 1024,
                    "memory_total_gb": float(parts[2]) / 1024,
                }
        except Exception:
            pass
        return None
    
    def get_k8s_resource_usage(self, namespace: str = "default") -> Dict[str, Any]:
        """Get Kubernetes resource usage."""
        try:
            result = subprocess.run(
                ["kubectl", "top", "pods", "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return {}
    
    def get_average_usage(self, last_n: int = 10) -> Dict[str, float]:
        """Get average resource usage from recent history."""
        if not self.history:
            return {}
        
        recent = self.history[-last_n:]
        
        return {
            "avg_cpu_percent": sum(u.cpu_percent for u in recent) / len(recent),
            "avg_memory_percent": sum(u.memory_percent for u in recent) / len(recent),
            "avg_gpu_percent": sum(u.gpu_percent or 0 for u in recent) / len(recent),
        }
