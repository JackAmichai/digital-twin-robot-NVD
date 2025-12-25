"""
Performance Profiling Framework for Digital Twin Robotics Lab.

Provides comprehensive profiling capabilities for:
- CPU profiling with cProfile, py-spy, and flame graphs
- GPU profiling with NVIDIA Nsight, nvprof, and CUDA events
- Memory profiling with tracemalloc and memory_profiler
- I/O profiling for disk and network operations
- Real-time performance monitoring and alerting
"""

import asyncio
import cProfile
import functools
import gc
import io
import json
import logging
import os
import pstats
import subprocess
import sys
import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ProfilerType(Enum):
    """Types of profilers available."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    LATENCY = "latency"
    COMBINED = "combined"


class GPUMetricType(Enum):
    """GPU metrics to collect."""
    UTILIZATION = "utilization"
    MEMORY_USED = "memory_used"
    MEMORY_FREE = "memory_free"
    TEMPERATURE = "temperature"
    POWER_DRAW = "power_draw"
    SM_CLOCK = "sm_clock"
    MEMORY_CLOCK = "memory_clock"
    TENSOR_UTILIZATION = "tensor_utilization"
    CUDA_CORES_ACTIVE = "cuda_cores_active"


@dataclass
class ProfileResult:
    """Result of a profiling session."""
    profiler_type: ProfilerType
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    metrics: Dict[str, Any]
    function_stats: Optional[List[Dict[str, Any]]] = None
    call_graph: Optional[Dict[str, Any]] = None
    memory_snapshots: Optional[List[Dict[str, Any]]] = None
    gpu_traces: Optional[List[Dict[str, Any]]] = None
    flame_graph_data: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass
class GPUProfile:
    """GPU profiling data."""
    device_id: int
    device_name: str
    compute_capability: Tuple[int, int]
    total_memory_mb: float
    metrics: Dict[GPUMetricType, List[Tuple[datetime, float]]]
    kernel_traces: List[Dict[str, Any]]
    memory_transfers: List[Dict[str, Any]]
    cuda_api_calls: List[Dict[str, Any]]


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    shared_mb: float
    heap_mb: float
    stack_mb: float
    gpu_memory_mb: Optional[float] = None
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    traceback: Optional[str] = None


@dataclass
class LatencyProfile:
    """Latency profiling for critical paths."""
    operation: str
    samples: List[float]  # Latency samples in ms
    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    std_dev: float
    min_latency: float
    max_latency: float


# =============================================================================
# Base Profiler Interface
# =============================================================================

class BaseProfiler(ABC):
    """Abstract base class for all profilers."""
    
    def __init__(self, name: str, output_dir: Optional[Path] = None):
        self.name = name
        self.output_dir = output_dir or Path("./profiling_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._is_running = False
        self._results: List[ProfileResult] = []
    
    @abstractmethod
    def start(self) -> None:
        """Start profiling."""
        pass
    
    @abstractmethod
    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        pass
    
    @abstractmethod
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics without stopping."""
        pass
    
    def export_results(self, result: ProfileResult, format: str = "json") -> Path:
        """Export results to file."""
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(self._result_to_dict(result), f, indent=2, default=str)
        elif format == "html":
            self._export_html(result, filepath)
        
        return filepath
    
    def _result_to_dict(self, result: ProfileResult) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "profiler_type": result.profiler_type.value,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "duration_seconds": result.duration_seconds,
            "metrics": result.metrics,
            "function_stats": result.function_stats,
            "recommendations": result.recommendations
        }
    
    def _export_html(self, result: ProfileResult, filepath: Path) -> None:
        """Export results as HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Profile Report - {self.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
                .warning {{ color: orange; }}
                .critical {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            </style>
        </head>
        <body>
            <h1>Performance Profile: {self.name}</h1>
            <p>Duration: {result.duration_seconds:.2f}s</p>
            <p>Time: {result.start_time} - {result.end_time}</p>
            
            <h2>Metrics</h2>
            <div class="metrics">
                {"".join(f'<div class="metric"><b>{k}:</b> {v}</div>' for k, v in result.metrics.items())}
            </div>
            
            <h2>Recommendations</h2>
            <ul>
                {"".join(f'<li>{r}</li>' for r in result.recommendations)}
            </ul>
        </body>
        </html>
        """
        with open(filepath, 'w') as f:
            f.write(html)


# =============================================================================
# CPU Profiler
# =============================================================================

class CPUProfiler(BaseProfiler):
    """
    CPU profiler with multiple backends:
    - cProfile for function-level profiling
    - py-spy for sampling-based profiling
    - line_profiler for line-by-line profiling
    """
    
    def __init__(
        self,
        name: str = "cpu_profiler",
        output_dir: Optional[Path] = None,
        use_cprofile: bool = True,
        sampling_interval_ms: int = 10
    ):
        super().__init__(name, output_dir)
        self.use_cprofile = use_cprofile
        self.sampling_interval_ms = sampling_interval_ms
        self._profiler: Optional[cProfile.Profile] = None
        self._start_time: Optional[datetime] = None
        self._samples: List[Dict[str, Any]] = []
    
    def start(self) -> None:
        """Start CPU profiling."""
        if self._is_running:
            raise RuntimeError("Profiler already running")
        
        self._is_running = True
        self._start_time = datetime.utcnow()
        self._samples = []
        
        if self.use_cprofile:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
        
        logger.info(f"CPU profiler '{self.name}' started")
    
    def stop(self) -> ProfileResult:
        """Stop CPU profiling and return results."""
        if not self._is_running:
            raise RuntimeError("Profiler not running")
        
        end_time = datetime.utcnow()
        
        if self.use_cprofile and self._profiler:
            self._profiler.disable()
        
        self._is_running = False
        
        # Process cProfile stats
        function_stats = self._process_cprofile_stats()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(function_stats)
        
        result = ProfileResult(
            profiler_type=ProfilerType.CPU,
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=(end_time - self._start_time).total_seconds(),
            metrics=self._calculate_metrics(function_stats),
            function_stats=function_stats,
            recommendations=recommendations
        )
        
        self._results.append(result)
        logger.info(f"CPU profiler '{self.name}' stopped")
        
        return result
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current CPU metrics."""
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        return {
            "cpu_percent_total": sum(cpu_percent) / len(cpu_percent),
            "cpu_percent_per_core": cpu_percent,
            "cpu_frequency_mhz": cpu_freq.current if cpu_freq else None,
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def _process_cprofile_stats(self) -> List[Dict[str, Any]]:
        """Process cProfile statistics into structured data."""
        if not self._profiler:
            return []
        
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        function_stats = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_number, func_name = func
            function_stats.append({
                "function": func_name,
                "filename": filename,
                "line_number": line_number,
                "call_count": nc,
                "recursive_call_count": cc,
                "total_time": tt,
                "cumulative_time": ct,
                "time_per_call": tt / nc if nc > 0 else 0,
                "cumulative_per_call": ct / nc if nc > 0 else 0
            })
        
        # Sort by cumulative time
        function_stats.sort(key=lambda x: x["cumulative_time"], reverse=True)
        
        return function_stats[:100]  # Top 100 functions
    
    def _calculate_metrics(self, function_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary metrics."""
        if not function_stats:
            return {}
        
        total_time = sum(f["total_time"] for f in function_stats)
        total_calls = sum(f["call_count"] for f in function_stats)
        
        return {
            "total_profiled_time_seconds": total_time,
            "total_function_calls": total_calls,
            "unique_functions_profiled": len(function_stats),
            "avg_time_per_call_ms": (total_time / total_calls * 1000) if total_calls > 0 else 0,
            "top_function_by_time": function_stats[0]["function"] if function_stats else None,
            "top_function_time_percent": (function_stats[0]["cumulative_time"] / total_time * 100) if total_time > 0 else 0
        }
    
    def _generate_recommendations(self, function_stats: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not function_stats:
            return recommendations
        
        # Check for hot functions
        total_time = sum(f["cumulative_time"] for f in function_stats)
        for func in function_stats[:5]:
            percent = (func["cumulative_time"] / total_time * 100) if total_time > 0 else 0
            if percent > 20:
                recommendations.append(
                    f"Function '{func['function']}' takes {percent:.1f}% of total time. "
                    f"Consider optimizing or caching results."
                )
        
        # Check for excessive function calls
        for func in function_stats:
            if func["call_count"] > 100000:
                recommendations.append(
                    f"Function '{func['function']}' called {func['call_count']:,} times. "
                    f"Consider reducing call frequency or batching."
                )
        
        return recommendations
    
    def generate_flame_graph(self, output_path: Optional[Path] = None) -> Path:
        """Generate flame graph SVG using py-spy or similar tool."""
        output_path = output_path or self.output_dir / f"{self.name}_flamegraph.svg"
        
        # Export stats for flame graph generation
        if self._profiler:
            stats_path = self.output_dir / f"{self.name}_stats.prof"
            self._profiler.dump_stats(str(stats_path))
            
            # Use flameprof or similar to generate flame graph
            try:
                subprocess.run([
                    "flameprof", str(stats_path),
                    "-o", str(output_path)
                ], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("flameprof not available, flame graph not generated")
        
        return output_path


# =============================================================================
# GPU Profiler
# =============================================================================

class GPUProfiler(BaseProfiler):
    """
    GPU profiler using NVIDIA tools:
    - nvidia-smi for utilization metrics
    - CUDA events for kernel timing
    - nvprof/Nsight for detailed traces
    - PyTorch profiler for deep learning workloads
    """
    
    def __init__(
        self,
        name: str = "gpu_profiler",
        output_dir: Optional[Path] = None,
        device_ids: Optional[List[int]] = None,
        sampling_interval_ms: int = 100,
        trace_cuda_kernels: bool = True
    ):
        super().__init__(name, output_dir)
        self.device_ids = device_ids or [0]
        self.sampling_interval_ms = sampling_interval_ms
        self.trace_cuda_kernels = trace_cuda_kernels
        self._start_time: Optional[datetime] = None
        self._metrics_history: Dict[int, Dict[GPUMetricType, List[Tuple[datetime, float]]]] = {}
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_sampling = threading.Event()
        
        # Check for GPU availability
        self._gpu_available = self._check_gpu_available()
        
        # Try to import PyTorch for CUDA profiling
        self._torch_available = False
        try:
            import torch
            self._torch_available = torch.cuda.is_available()
        except ImportError:
            pass
    
    def _check_gpu_available(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        """Start GPU profiling."""
        if self._is_running:
            raise RuntimeError("Profiler already running")
        
        if not self._gpu_available:
            logger.warning("No NVIDIA GPU detected, GPU profiling limited")
        
        self._is_running = True
        self._start_time = datetime.utcnow()
        self._metrics_history = {
            device_id: {metric: [] for metric in GPUMetricType}
            for device_id in self.device_ids
        }
        
        # Start background sampling thread
        self._stop_sampling.clear()
        self._sampling_thread = threading.Thread(target=self._sample_metrics, daemon=True)
        self._sampling_thread.start()
        
        logger.info(f"GPU profiler '{self.name}' started")
    
    def stop(self) -> ProfileResult:
        """Stop GPU profiling and return results."""
        if not self._is_running:
            raise RuntimeError("Profiler not running")
        
        end_time = datetime.utcnow()
        
        # Stop sampling thread
        self._stop_sampling.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=5)
        
        self._is_running = False
        
        # Process metrics
        metrics = self._calculate_gpu_metrics()
        recommendations = self._generate_gpu_recommendations(metrics)
        
        result = ProfileResult(
            profiler_type=ProfilerType.GPU,
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=(end_time - self._start_time).total_seconds(),
            metrics=metrics,
            gpu_traces=self._get_gpu_traces(),
            recommendations=recommendations
        )
        
        self._results.append(result)
        logger.info(f"GPU profiler '{self.name}' stopped")
        
        return result
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics."""
        if not self._gpu_available:
            return {"error": "No GPU available"}
        
        metrics = {}
        for device_id in self.device_ids:
            gpu_metrics = self._query_nvidia_smi(device_id)
            metrics[f"gpu_{device_id}"] = gpu_metrics
        
        return metrics
    
    def _sample_metrics(self) -> None:
        """Background thread for sampling GPU metrics."""
        while not self._stop_sampling.is_set():
            timestamp = datetime.utcnow()
            
            for device_id in self.device_ids:
                try:
                    gpu_metrics = self._query_nvidia_smi(device_id)
                    
                    if gpu_metrics:
                        history = self._metrics_history[device_id]
                        history[GPUMetricType.UTILIZATION].append(
                            (timestamp, gpu_metrics.get("utilization_gpu", 0))
                        )
                        history[GPUMetricType.MEMORY_USED].append(
                            (timestamp, gpu_metrics.get("memory_used_mb", 0))
                        )
                        history[GPUMetricType.TEMPERATURE].append(
                            (timestamp, gpu_metrics.get("temperature_c", 0))
                        )
                        history[GPUMetricType.POWER_DRAW].append(
                            (timestamp, gpu_metrics.get("power_draw_w", 0))
                        )
                except Exception as e:
                    logger.error(f"Error sampling GPU metrics: {e}")
            
            time.sleep(self.sampling_interval_ms / 1000)
    
    def _query_nvidia_smi(self, device_id: int) -> Dict[str, Any]:
        """Query nvidia-smi for GPU metrics."""
        try:
            result = subprocess.run([
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return {}
            
            values = result.stdout.strip().split(", ")
            if len(values) >= 10:
                return {
                    "name": values[0],
                    "utilization_gpu": float(values[1]) if values[1] != "[N/A]" else 0,
                    "utilization_memory": float(values[2]) if values[2] != "[N/A]" else 0,
                    "memory_total_mb": float(values[3]) if values[3] != "[N/A]" else 0,
                    "memory_used_mb": float(values[4]) if values[4] != "[N/A]" else 0,
                    "memory_free_mb": float(values[5]) if values[5] != "[N/A]" else 0,
                    "temperature_c": float(values[6]) if values[6] != "[N/A]" else 0,
                    "power_draw_w": float(values[7]) if values[7] != "[N/A]" else 0,
                    "sm_clock_mhz": float(values[8]) if values[8] != "[N/A]" else 0,
                    "memory_clock_mhz": float(values[9]) if values[9] != "[N/A]" else 0
                }
        except Exception as e:
            logger.error(f"nvidia-smi query failed: {e}")
        
        return {}
    
    def _calculate_gpu_metrics(self) -> Dict[str, Any]:
        """Calculate summary GPU metrics."""
        metrics = {}
        
        for device_id, history in self._metrics_history.items():
            util_samples = [v for _, v in history[GPUMetricType.UTILIZATION]]
            mem_samples = [v for _, v in history[GPUMetricType.MEMORY_USED]]
            temp_samples = [v for _, v in history[GPUMetricType.TEMPERATURE]]
            power_samples = [v for _, v in history[GPUMetricType.POWER_DRAW]]
            
            metrics[f"gpu_{device_id}"] = {
                "utilization_avg_percent": sum(util_samples) / len(util_samples) if util_samples else 0,
                "utilization_max_percent": max(util_samples) if util_samples else 0,
                "memory_used_avg_mb": sum(mem_samples) / len(mem_samples) if mem_samples else 0,
                "memory_used_max_mb": max(mem_samples) if mem_samples else 0,
                "temperature_avg_c": sum(temp_samples) / len(temp_samples) if temp_samples else 0,
                "temperature_max_c": max(temp_samples) if temp_samples else 0,
                "power_draw_avg_w": sum(power_samples) / len(power_samples) if power_samples else 0,
                "power_draw_max_w": max(power_samples) if power_samples else 0,
                "sample_count": len(util_samples)
            }
        
        return metrics
    
    def _get_gpu_traces(self) -> List[Dict[str, Any]]:
        """Get GPU kernel traces if available."""
        # This would integrate with CUDA profiling tools
        # Placeholder for CUDA event traces
        return []
    
    def _generate_gpu_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate GPU optimization recommendations."""
        recommendations = []
        
        for device_id in self.device_ids:
            gpu_metrics = metrics.get(f"gpu_{device_id}", {})
            
            # Low utilization warning
            util_avg = gpu_metrics.get("utilization_avg_percent", 0)
            if util_avg < 50:
                recommendations.append(
                    f"GPU {device_id} average utilization is {util_avg:.1f}%. "
                    f"Consider batching more work or using smaller models."
                )
            
            # High memory usage
            mem_max = gpu_metrics.get("memory_used_max_mb", 0)
            if mem_max > 0:
                # Estimate total memory (would need to query)
                recommendations.append(
                    f"GPU {device_id} peak memory usage: {mem_max:.0f} MB. "
                    f"Monitor for OOM risks during inference spikes."
                )
            
            # High temperature warning
            temp_max = gpu_metrics.get("temperature_max_c", 0)
            if temp_max > 80:
                recommendations.append(
                    f"GPU {device_id} reached {temp_max}°C. "
                    f"Check cooling and consider throttling workload."
                )
        
        return recommendations


# =============================================================================
# Memory Profiler
# =============================================================================

class MemoryProfiler(BaseProfiler):
    """
    Memory profiler for tracking allocations and leaks:
    - tracemalloc for Python allocation tracking
    - psutil for system memory
    - GPU memory tracking via CUDA
    """
    
    def __init__(
        self,
        name: str = "memory_profiler",
        output_dir: Optional[Path] = None,
        track_gpu_memory: bool = True,
        snapshot_interval_ms: int = 1000,
        top_allocations: int = 20
    ):
        super().__init__(name, output_dir)
        self.track_gpu_memory = track_gpu_memory
        self.snapshot_interval_ms = snapshot_interval_ms
        self.top_allocations = top_allocations
        self._start_time: Optional[datetime] = None
        self._snapshots: List[MemorySnapshot] = []
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_sampling = threading.Event()
    
    def start(self) -> None:
        """Start memory profiling."""
        if self._is_running:
            raise RuntimeError("Profiler already running")
        
        self._is_running = True
        self._start_time = datetime.utcnow()
        self._snapshots = []
        
        # Start tracemalloc
        tracemalloc.start()
        
        # Start background snapshot thread
        self._stop_sampling.clear()
        self._sampling_thread = threading.Thread(target=self._take_snapshots, daemon=True)
        self._sampling_thread.start()
        
        logger.info(f"Memory profiler '{self.name}' started")
    
    def stop(self) -> ProfileResult:
        """Stop memory profiling and return results."""
        if not self._is_running:
            raise RuntimeError("Profiler not running")
        
        end_time = datetime.utcnow()
        
        # Stop sampling thread
        self._stop_sampling.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=5)
        
        # Take final snapshot
        self._capture_snapshot()
        
        # Stop tracemalloc
        tracemalloc.stop()
        
        self._is_running = False
        
        # Calculate metrics
        metrics = self._calculate_memory_metrics()
        recommendations = self._generate_memory_recommendations(metrics)
        
        result = ProfileResult(
            profiler_type=ProfilerType.MEMORY,
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=(end_time - self._start_time).total_seconds(),
            metrics=metrics,
            memory_snapshots=[self._snapshot_to_dict(s) for s in self._snapshots],
            recommendations=recommendations
        )
        
        self._results.append(result)
        logger.info(f"Memory profiler '{self.name}' stopped")
        
        return result
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current memory metrics."""
        import psutil
        
        process = psutil.Process()
        mem_info = process.memory_info()
        
        metrics = {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "system_total_mb": psutil.virtual_memory().total / (1024 * 1024),
            "system_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "system_percent": psutil.virtual_memory().percent
        }
        
        # Add tracemalloc info if running
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            metrics["tracemalloc_current_mb"] = current / (1024 * 1024)
            metrics["tracemalloc_peak_mb"] = peak / (1024 * 1024)
        
        return metrics
    
    def _take_snapshots(self) -> None:
        """Background thread for taking memory snapshots."""
        while not self._stop_sampling.is_set():
            self._capture_snapshot()
            time.sleep(self.snapshot_interval_ms / 1000)
    
    def _capture_snapshot(self) -> None:
        """Capture a memory snapshot."""
        import psutil
        
        timestamp = datetime.utcnow()
        process = psutil.Process()
        mem_info = process.memory_info()
        
        # Get top allocations from tracemalloc
        top_allocations = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:self.top_allocations]
            
            for stat in top_stats:
                top_allocations.append({
                    "file": str(stat.traceback),
                    "size_kb": stat.size / 1024,
                    "count": stat.count
                })
        
        # Get GPU memory if available
        gpu_memory_mb = None
        if self.track_gpu_memory:
            try:
                result = subprocess.run([
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    current_pid = os.getpid()
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(', ')
                            if len(parts) >= 2 and int(parts[0]) == current_pid:
                                gpu_memory_mb = float(parts[1])
                                break
            except Exception:
                pass
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            shared_mb=getattr(mem_info, 'shared', 0) / (1024 * 1024),
            heap_mb=0,  # Would require more detailed analysis
            stack_mb=0,
            gpu_memory_mb=gpu_memory_mb,
            top_allocations=top_allocations
        )
        
        self._snapshots.append(snapshot)
    
    def _snapshot_to_dict(self, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "rss_mb": snapshot.rss_mb,
            "vms_mb": snapshot.vms_mb,
            "shared_mb": snapshot.shared_mb,
            "gpu_memory_mb": snapshot.gpu_memory_mb,
            "top_allocations": snapshot.top_allocations
        }
    
    def _calculate_memory_metrics(self) -> Dict[str, Any]:
        """Calculate summary memory metrics."""
        if not self._snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self._snapshots]
        vms_values = [s.vms_mb for s in self._snapshots]
        
        metrics = {
            "rss_start_mb": rss_values[0],
            "rss_end_mb": rss_values[-1],
            "rss_max_mb": max(rss_values),
            "rss_growth_mb": rss_values[-1] - rss_values[0],
            "vms_max_mb": max(vms_values),
            "snapshot_count": len(self._snapshots)
        }
        
        # Check for GPU memory
        gpu_values = [s.gpu_memory_mb for s in self._snapshots if s.gpu_memory_mb is not None]
        if gpu_values:
            metrics["gpu_memory_max_mb"] = max(gpu_values)
            metrics["gpu_memory_avg_mb"] = sum(gpu_values) / len(gpu_values)
        
        return metrics
    
    def _generate_memory_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Check for memory growth (potential leak)
        growth = metrics.get("rss_growth_mb", 0)
        if growth > 100:
            recommendations.append(
                f"Memory grew by {growth:.1f} MB during profiling. "
                f"Check for memory leaks or unbounded caches."
            )
        
        # High memory usage
        max_rss = metrics.get("rss_max_mb", 0)
        if max_rss > 4096:
            recommendations.append(
                f"Peak memory usage: {max_rss:.0f} MB. "
                f"Consider optimizing data structures or using generators."
            )
        
        # GPU memory
        gpu_max = metrics.get("gpu_memory_max_mb", 0)
        if gpu_max > 8000:
            recommendations.append(
                f"GPU memory peak: {gpu_max:.0f} MB. "
                f"Consider gradient checkpointing or model parallelism."
            )
        
        return recommendations


# =============================================================================
# Latency Profiler
# =============================================================================

class LatencyProfiler(BaseProfiler):
    """
    Latency profiler for critical path analysis:
    - End-to-end latency tracking
    - Percentile calculations
    - SLA monitoring
    """
    
    def __init__(
        self,
        name: str = "latency_profiler",
        output_dir: Optional[Path] = None,
        percentiles: List[float] = None
    ):
        super().__init__(name, output_dir)
        self.percentiles = percentiles or [50, 90, 95, 99]
        self._start_time: Optional[datetime] = None
        self._latencies: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start latency profiling."""
        if self._is_running:
            raise RuntimeError("Profiler already running")
        
        self._is_running = True
        self._start_time = datetime.utcnow()
        self._latencies = {}
        
        logger.info(f"Latency profiler '{self.name}' started")
    
    def stop(self) -> ProfileResult:
        """Stop latency profiling and return results."""
        if not self._is_running:
            raise RuntimeError("Profiler not running")
        
        end_time = datetime.utcnow()
        self._is_running = False
        
        # Calculate metrics for each operation
        metrics = {}
        function_stats = []
        
        for operation, samples in self._latencies.items():
            if samples:
                stats = self._calculate_latency_stats(operation, samples)
                metrics[operation] = stats
                function_stats.append({
                    "operation": operation,
                    **stats
                })
        
        recommendations = self._generate_latency_recommendations(metrics)
        
        result = ProfileResult(
            profiler_type=ProfilerType.LATENCY,
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=(end_time - self._start_time).total_seconds(),
            metrics=metrics,
            function_stats=function_stats,
            recommendations=recommendations
        )
        
        self._results.append(result)
        logger.info(f"Latency profiler '{self.name}' stopped")
        
        return result
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current latency metrics."""
        with self._lock:
            return {
                op: self._calculate_latency_stats(op, samples)
                for op, samples in self._latencies.items()
                if samples
            }
    
    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record a latency sample for an operation."""
        with self._lock:
            if operation not in self._latencies:
                self._latencies[operation] = []
            self._latencies[operation].append(latency_ms)
    
    @contextmanager
    def measure(self, operation: str):
        """Context manager for measuring operation latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.record_latency(operation, latency_ms)
    
    def _calculate_latency_stats(self, operation: str, samples: List[float]) -> Dict[str, float]:
        """Calculate latency statistics."""
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        def percentile(p: float) -> float:
            k = (n - 1) * (p / 100)
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])
        
        mean = sum(samples) / n
        variance = sum((x - mean) ** 2 for x in samples) / n
        std_dev = variance ** 0.5
        
        return {
            "sample_count": n,
            "min_ms": sorted_samples[0],
            "max_ms": sorted_samples[-1],
            "mean_ms": mean,
            "std_dev_ms": std_dev,
            "p50_ms": percentile(50),
            "p90_ms": percentile(90),
            "p95_ms": percentile(95),
            "p99_ms": percentile(99)
        }
    
    def _generate_latency_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate latency optimization recommendations."""
        recommendations = []
        
        for operation, stats in metrics.items():
            # High P99 latency
            p99 = stats.get("p99_ms", 0)
            mean = stats.get("mean_ms", 0)
            
            if p99 > 1000:  # > 1 second
                recommendations.append(
                    f"Operation '{operation}' P99 latency is {p99:.0f}ms. "
                    f"Consider async processing or caching."
                )
            
            # High variance (P99 >> mean)
            if mean > 0 and p99 / mean > 10:
                recommendations.append(
                    f"Operation '{operation}' has high latency variance "
                    f"(P99={p99:.0f}ms vs mean={mean:.0f}ms). "
                    f"Investigate intermittent slowdowns."
                )
        
        return recommendations


# =============================================================================
# Combined Profiler
# =============================================================================

class CombinedProfiler:
    """
    Orchestrates multiple profilers for comprehensive analysis.
    """
    
    def __init__(
        self,
        name: str = "combined_profiler",
        output_dir: Optional[Path] = None,
        enable_cpu: bool = True,
        enable_gpu: bool = True,
        enable_memory: bool = True,
        enable_latency: bool = True
    ):
        self.name = name
        self.output_dir = output_dir or Path("./profiling_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profilers: Dict[str, BaseProfiler] = {}
        
        if enable_cpu:
            self.profilers["cpu"] = CPUProfiler(f"{name}_cpu", self.output_dir)
        if enable_gpu:
            self.profilers["gpu"] = GPUProfiler(f"{name}_gpu", self.output_dir)
        if enable_memory:
            self.profilers["memory"] = MemoryProfiler(f"{name}_memory", self.output_dir)
        if enable_latency:
            self.profilers["latency"] = LatencyProfiler(f"{name}_latency", self.output_dir)
        
        self._is_running = False
    
    def start(self) -> None:
        """Start all profilers."""
        if self._is_running:
            raise RuntimeError("Profilers already running")
        
        for profiler in self.profilers.values():
            profiler.start()
        
        self._is_running = True
        logger.info(f"Combined profiler '{self.name}' started with {list(self.profilers.keys())}")
    
    def stop(self) -> Dict[str, ProfileResult]:
        """Stop all profilers and return results."""
        if not self._is_running:
            raise RuntimeError("Profilers not running")
        
        results = {}
        for name, profiler in self.profilers.items():
            results[name] = profiler.stop()
        
        self._is_running = False
        logger.info(f"Combined profiler '{self.name}' stopped")
        
        return results
    
    def get_current_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics from all profilers."""
        return {
            name: profiler.get_current_metrics()
            for name, profiler in self.profilers.items()
        }
    
    def export_report(self, results: Dict[str, ProfileResult]) -> Path:
        """Export combined HTML report."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{self.name}_report_{timestamp}.html"
        
        sections = []
        all_recommendations = []
        
        for name, result in results.items():
            all_recommendations.extend(result.recommendations)
            
            metrics_html = "".join(
                f'<tr><td>{k}</td><td>{v}</td></tr>'
                for k, v in result.metrics.items()
            )
            
            sections.append(f"""
                <h2>{name.upper()} Profile</h2>
                <p>Duration: {result.duration_seconds:.2f}s</p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {metrics_html}
                </table>
            """)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {self.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f5f5f5; }}
                .recommendation {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
            </style>
        </head>
        <body>
            <h1>Performance Report: {self.name}</h1>
            <p>Generated: {datetime.utcnow().isoformat()}</p>
            
            <h2>Recommendations</h2>
            {"".join(f'<div class="recommendation">{r}</div>' for r in all_recommendations)}
            
            {"".join(sections)}
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        return filepath


# =============================================================================
# Decorators for Easy Profiling
# =============================================================================

def profile_function(profiler_type: ProfilerType = ProfilerType.CPU):
    """Decorator to profile a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profiler_type == ProfilerType.CPU:
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                    # Log stats
                    stream = io.StringIO()
                    stats = pstats.Stats(profiler, stream=stream)
                    stats.sort_stats('cumulative')
                    stats.print_stats(10)
                    logger.debug(f"Profile for {func.__name__}:\n{stream.getvalue()}")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_async_function(profiler_type: ProfilerType = ProfilerType.LATENCY):
    """Decorator to profile an async function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"{func.__name__} completed in {duration_ms:.2f}ms")
        return wrapper
    return decorator


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Combined profiling session
    profiler = CombinedProfiler(
        name="example_session",
        enable_cpu=True,
        enable_gpu=True,
        enable_memory=True,
        enable_latency=True
    )
    
    profiler.start()
    
    # Simulate some work
    latency_profiler = profiler.profilers.get("latency")
    if latency_profiler:
        for i in range(100):
            with latency_profiler.measure("example_operation"):
                time.sleep(0.01)  # Simulate work
    
    results = profiler.stop()
    
    # Export report
    report_path = profiler.export_report(results)
    print(f"Report exported to: {report_path}")
