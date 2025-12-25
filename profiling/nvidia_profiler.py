"""
NVIDIA-specific profiling tools for GPU workloads.

Provides integration with:
- NVIDIA Nsight Systems for system-wide profiling
- NVIDIA Nsight Compute for kernel analysis
- CUDA Events for timing
- PyTorch Profiler for deep learning
- TensorRT Profiler for inference
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CUDA Event Timer
# =============================================================================

class CUDAEventTimer:
    """
    High-precision GPU timing using CUDA events.
    
    CUDA events provide the most accurate way to measure GPU kernel execution
    time because they're recorded directly on the GPU timeline.
    
    Example:
        timer = CUDAEventTimer()
        timer.start()
        # GPU operations here
        elapsed_ms = timer.stop()
    """
    
    def __init__(self):
        self._torch_available = False
        self._start_event = None
        self._end_event = None
        
        try:
            import torch
            if torch.cuda.is_available():
                self._torch_available = True
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
        except ImportError:
            logger.warning("PyTorch not available, CUDA event timing disabled")
    
    def start(self) -> None:
        """Record start event on GPU."""
        if self._torch_available and self._start_event:
            self._start_event.record()
    
    def stop(self) -> float:
        """Record end event and return elapsed time in milliseconds."""
        if self._torch_available and self._end_event:
            self._end_event.record()
            self._end_event.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        return 0.0
    
    @contextmanager
    def measure(self):
        """Context manager for timing GPU operations."""
        self.start()
        try:
            yield self
        finally:
            pass  # Elapsed time available via stop()


# =============================================================================
# Nsight Systems Integration
# =============================================================================

@dataclass
class NsightSystemsConfig:
    """Configuration for Nsight Systems profiling."""
    output_dir: Path = field(default_factory=lambda: Path("./nsight_reports"))
    trace_cuda: bool = True
    trace_nvtx: bool = True
    trace_osrt: bool = True  # OS Runtime
    trace_cudnn: bool = True
    trace_cublas: bool = True
    sample_cpu: bool = True
    cpu_sampling_frequency: int = 1000  # Hz
    gpu_metrics: bool = True
    duration_seconds: Optional[int] = None
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


class NsightSystemsProfiler:
    """
    Integration with NVIDIA Nsight Systems for system-wide profiling.
    
    Nsight Systems provides a holistic view of GPU and CPU activity,
    including CUDA API calls, kernel launches, memory transfers,
    and CPU-GPU synchronization.
    
    Prerequisites:
        - NVIDIA Nsight Systems installed (nsys command available)
        - NVIDIA GPU with supported driver
    
    Example:
        profiler = NsightSystemsProfiler()
        report_path = profiler.profile_command(
            ["python", "inference.py"],
            duration_seconds=30
        )
    """
    
    def __init__(self, config: Optional[NsightSystemsConfig] = None):
        self.config = config or NsightSystemsConfig()
        self._nsys_available = self._check_nsys()
    
    def _check_nsys(self) -> bool:
        """Check if nsys is available."""
        try:
            result = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Nsight Systems available: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        logger.warning("Nsight Systems (nsys) not found in PATH")
        return False
    
    def profile_command(
        self,
        command: List[str],
        output_name: Optional[str] = None,
        duration_seconds: Optional[int] = None
    ) -> Optional[Path]:
        """
        Profile a command using Nsight Systems.
        
        Args:
            command: Command and arguments to profile
            output_name: Base name for output report
            duration_seconds: Maximum profiling duration
        
        Returns:
            Path to the generated report file (.nsys-rep)
        """
        if not self._nsys_available:
            logger.error("Nsight Systems not available")
            return None
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_name = output_name or f"profile_{timestamp}"
        output_path = self.config.output_dir / output_name
        
        # Build nsys command
        nsys_cmd = [
            "nsys", "profile",
            "-o", str(output_path),
            "--force-overwrite=true",
            "--export=sqlite,json"
        ]
        
        # Add trace options
        if self.config.trace_cuda:
            nsys_cmd.extend(["--trace=cuda"])
        if self.config.trace_nvtx:
            nsys_cmd.extend(["--trace=nvtx"])
        if self.config.trace_osrt:
            nsys_cmd.extend(["--trace=osrt"])
        if self.config.trace_cudnn:
            nsys_cmd.extend(["--trace=cudnn"])
        if self.config.trace_cublas:
            nsys_cmd.extend(["--trace=cublas"])
        
        # CPU sampling
        if self.config.sample_cpu:
            nsys_cmd.extend([
                "--sample=cpu",
                f"--sampling-frequency={self.config.cpu_sampling_frequency}"
            ])
        
        # GPU metrics
        if self.config.gpu_metrics:
            nsys_cmd.extend(["--gpu-metrics-device=all"])
        
        # Duration limit
        duration = duration_seconds or self.config.duration_seconds
        if duration:
            nsys_cmd.extend([f"--duration={duration}"])
        
        # Add the command to profile
        nsys_cmd.extend(command)
        
        logger.info(f"Running Nsight Systems: {' '.join(nsys_cmd)}")
        
        try:
            result = subprocess.run(
                nsys_cmd,
                capture_output=True,
                text=True,
                timeout=duration + 60 if duration else 3600
            )
            
            if result.returncode == 0:
                report_path = output_path.with_suffix(".nsys-rep")
                logger.info(f"Nsight report generated: {report_path}")
                return report_path
            else:
                logger.error(f"Nsight profiling failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Nsight profiling timed out")
            return None
        except Exception as e:
            logger.error(f"Nsight profiling error: {e}")
            return None
    
    def analyze_report(self, report_path: Path) -> Dict[str, Any]:
        """
        Analyze a Nsight Systems report and extract key metrics.
        
        Args:
            report_path: Path to .nsys-rep file
        
        Returns:
            Dictionary with analysis results
        """
        if not report_path.exists():
            return {"error": "Report file not found"}
        
        # Export to JSON for analysis
        json_path = report_path.with_suffix(".json")
        
        if not json_path.exists():
            try:
                subprocess.run([
                    "nsys", "export",
                    "-t", "json",
                    str(report_path)
                ], check=True)
            except Exception as e:
                logger.error(f"Failed to export report: {e}")
                return {"error": str(e)}
        
        # Parse JSON report
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            return self._extract_metrics(data)
        except Exception as e:
            logger.error(f"Failed to parse report: {e}")
            return {"error": str(e)}
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from Nsight JSON data."""
        metrics = {
            "cuda_api_calls": 0,
            "kernel_launches": 0,
            "memory_operations": 0,
            "total_kernel_time_ms": 0,
            "total_memcpy_time_ms": 0,
            "gpu_utilization_percent": 0,
            "top_kernels": [],
            "memory_transfers": []
        }
        
        # Process CUDA API calls
        if "cuda_api" in data:
            metrics["cuda_api_calls"] = len(data["cuda_api"])
        
        # Process kernel data
        if "cuda_gpu_kernel" in data:
            kernels = data["cuda_gpu_kernel"]
            metrics["kernel_launches"] = len(kernels)
            
            # Aggregate kernel times
            kernel_times = {}
            for kernel in kernels:
                name = kernel.get("name", "unknown")
                duration_ns = kernel.get("duration", 0)
                
                if name not in kernel_times:
                    kernel_times[name] = {"count": 0, "total_ns": 0}
                kernel_times[name]["count"] += 1
                kernel_times[name]["total_ns"] += duration_ns
            
            # Top kernels by time
            sorted_kernels = sorted(
                kernel_times.items(),
                key=lambda x: x[1]["total_ns"],
                reverse=True
            )[:10]
            
            metrics["top_kernels"] = [
                {
                    "name": name,
                    "count": stats["count"],
                    "total_ms": stats["total_ns"] / 1e6
                }
                for name, stats in sorted_kernels
            ]
            
            metrics["total_kernel_time_ms"] = sum(
                k.get("duration", 0) for k in kernels
            ) / 1e6
        
        return metrics


# =============================================================================
# PyTorch Profiler Integration
# =============================================================================

class PyTorchProfiler:
    """
    Integration with PyTorch's built-in profiler for deep learning workloads.
    
    PyTorch Profiler provides detailed insights into:
    - Operator-level timing
    - CUDA kernel execution
    - Memory allocation patterns
    - TensorBoard integration
    
    Example:
        profiler = PyTorchProfiler()
        with profiler.profile() as prof:
            model(input_tensor)
        profiler.export_chrome_trace("trace.json")
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True
    ):
        self.output_dir = output_dir or Path("./pytorch_profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        
        self._torch_available = False
        self._profiler = None
        
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity
            self._torch_available = True
            self._torch = torch
            self._profile = profile
            self._ProfilerActivity = ProfilerActivity
        except ImportError:
            logger.warning("PyTorch not available")
    
    @contextmanager
    def profile(
        self,
        activities: Optional[List[str]] = None,
        schedule: Optional[Dict[str, int]] = None
    ):
        """
        Context manager for PyTorch profiling.
        
        Args:
            activities: List of activities to profile ["cpu", "cuda"]
            schedule: Profiling schedule {"wait": 1, "warmup": 1, "active": 3}
        
        Yields:
            PyTorch profiler object
        """
        if not self._torch_available:
            yield None
            return
        
        # Determine activities
        activity_list = []
        activities = activities or ["cpu", "cuda"]
        
        if "cpu" in activities:
            activity_list.append(self._ProfilerActivity.CPU)
        if "cuda" in activities and self._torch.cuda.is_available():
            activity_list.append(self._ProfilerActivity.CUDA)
        
        # Build profiler kwargs
        profiler_kwargs = {
            "activities": activity_list,
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_stack": self.with_stack,
            "with_flops": self.with_flops
        }
        
        # Add schedule if provided
        if schedule:
            from torch.profiler import schedule as torch_schedule
            profiler_kwargs["schedule"] = torch_schedule(
                wait=schedule.get("wait", 1),
                warmup=schedule.get("warmup", 1),
                active=schedule.get("active", 3),
                repeat=schedule.get("repeat", 1)
            )
        
        with self._profile(**profiler_kwargs) as prof:
            self._profiler = prof
            yield prof
    
    def export_chrome_trace(self, filename: Optional[str] = None) -> Path:
        """Export trace in Chrome trace format for visualization."""
        if not self._profiler:
            raise RuntimeError("No profiler data available")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"trace_{timestamp}.json"
        filepath = self.output_dir / filename
        
        self._profiler.export_chrome_trace(str(filepath))
        logger.info(f"Chrome trace exported to: {filepath}")
        
        return filepath
    
    def export_stacks(self, filename: Optional[str] = None) -> Path:
        """Export stack traces for flame graph generation."""
        if not self._profiler:
            raise RuntimeError("No profiler data available")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"stacks_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        self._profiler.export_stacks(str(filepath))
        logger.info(f"Stack traces exported to: {filepath}")
        
        return filepath
    
    def get_key_averages(self) -> List[Dict[str, Any]]:
        """Get averaged statistics for all operations."""
        if not self._profiler:
            return []
        
        key_averages = self._profiler.key_averages()
        
        results = []
        for item in key_averages:
            results.append({
                "name": item.key,
                "cpu_time_total_us": item.cpu_time_total,
                "cuda_time_total_us": item.cuda_time_total,
                "cpu_time_avg_us": item.cpu_time_total / item.count if item.count > 0 else 0,
                "cuda_time_avg_us": item.cuda_time_total / item.count if item.count > 0 else 0,
                "count": item.count,
                "cpu_memory_usage": item.cpu_memory_usage,
                "cuda_memory_usage": getattr(item, 'cuda_memory_usage', 0),
                "flops": getattr(item, 'flops', 0)
            })
        
        # Sort by CUDA time (most relevant for GPU workloads)
        results.sort(key=lambda x: x["cuda_time_total_us"], reverse=True)
        
        return results
    
    def print_summary(self, sort_by: str = "cuda_time_total", top_n: int = 20) -> str:
        """Print a formatted summary of profiling results."""
        if not self._profiler:
            return "No profiler data available"
        
        return self._profiler.key_averages().table(
            sort_by=sort_by,
            row_limit=top_n
        )


# =============================================================================
# TensorRT Profiler
# =============================================================================

class TensorRTProfiler:
    """
    Profiler for TensorRT inference optimization.
    
    Analyzes TensorRT engine performance including:
    - Layer-wise execution times
    - Memory allocation
    - Optimization opportunities
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./tensorrt_profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._trt_available = False
        try:
            import tensorrt as trt
            self._trt_available = True
            self._trt = trt
        except ImportError:
            logger.warning("TensorRT not available")
    
    def profile_engine(
        self,
        engine_path: Path,
        input_shapes: Dict[str, Tuple[int, ...]],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Profile a TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine file
            input_shapes: Dictionary of input name -> shape
            num_iterations: Number of inference iterations
            warmup_iterations: Warmup iterations before timing
        
        Returns:
            Profiling results dictionary
        """
        if not self._trt_available:
            return {"error": "TensorRT not available"}
        
        # Use trtexec for profiling (more reliable than Python API)
        trtexec_cmd = [
            "trtexec",
            f"--loadEngine={engine_path}",
            f"--iterations={num_iterations}",
            f"--warmUp={warmup_iterations * 1000}",  # ms
            "--dumpProfile",
            "--separateProfileRun",
            "--verbose"
        ]
        
        # Add input shapes
        for name, shape in input_shapes.items():
            shape_str = "x".join(map(str, shape))
            trtexec_cmd.append(f"--shapes={name}:{shape_str}")
        
        try:
            result = subprocess.run(
                trtexec_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return self._parse_trtexec_output(result.stdout)
            
        except FileNotFoundError:
            logger.error("trtexec not found in PATH")
            return {"error": "trtexec not available"}
        except Exception as e:
            logger.error(f"TensorRT profiling failed: {e}")
            return {"error": str(e)}
    
    def _parse_trtexec_output(self, output: str) -> Dict[str, Any]:
        """Parse trtexec output for metrics."""
        metrics = {
            "throughput_qps": None,
            "latency_mean_ms": None,
            "latency_p99_ms": None,
            "gpu_compute_time_ms": None,
            "host_latency_ms": None,
            "layer_times": []
        }
        
        for line in output.split('\n'):
            # Parse throughput
            if "Throughput:" in line:
                try:
                    metrics["throughput_qps"] = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
            
            # Parse latency
            if "mean:" in line.lower():
                try:
                    metrics["latency_mean_ms"] = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
            
            if "99%" in line:
                try:
                    metrics["latency_p99_ms"] = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
        
        return metrics


# =============================================================================
# GPU Memory Analyzer
# =============================================================================

class GPUMemoryAnalyzer:
    """
    Detailed GPU memory analysis for debugging OOM issues.
    
    Tracks:
    - Memory allocations by tensor
    - Memory fragmentation
    - Peak memory usage
    - Memory leaks
    """
    
    def __init__(self):
        self._torch_available = False
        try:
            import torch
            self._torch_available = torch.cuda.is_available()
            self._torch = torch
        except ImportError:
            pass
        
        self._snapshots: List[Dict[str, Any]] = []
    
    def get_memory_summary(self, device: int = 0) -> Dict[str, Any]:
        """Get current GPU memory summary."""
        if not self._torch_available:
            return {"error": "CUDA not available"}
        
        self._torch.cuda.set_device(device)
        
        return {
            "allocated_mb": self._torch.cuda.memory_allocated(device) / (1024 * 1024),
            "reserved_mb": self._torch.cuda.memory_reserved(device) / (1024 * 1024),
            "max_allocated_mb": self._torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "max_reserved_mb": self._torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        }
    
    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot for later comparison."""
        if not self._torch_available:
            return {}
        
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            **self.get_memory_summary()
        }
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def compare_snapshots(
        self,
        snapshot1_idx: int = -2,
        snapshot2_idx: int = -1
    ) -> Dict[str, Any]:
        """Compare two snapshots to detect memory changes."""
        if len(self._snapshots) < 2:
            return {"error": "Need at least 2 snapshots"}
        
        s1 = self._snapshots[snapshot1_idx]
        s2 = self._snapshots[snapshot2_idx]
        
        return {
            "from_label": s1["label"],
            "to_label": s2["label"],
            "allocated_delta_mb": s2["allocated_mb"] - s1["allocated_mb"],
            "reserved_delta_mb": s2["reserved_mb"] - s1["reserved_mb"],
            "time_delta_seconds": (
                datetime.fromisoformat(s2["timestamp"]) -
                datetime.fromisoformat(s1["timestamp"])
            ).total_seconds()
        }
    
    def find_memory_leaks(self) -> List[Dict[str, Any]]:
        """Analyze snapshots to find potential memory leaks."""
        if len(self._snapshots) < 3:
            return []
        
        leaks = []
        
        # Look for consistently increasing memory
        allocated_values = [s["allocated_mb"] for s in self._snapshots]
        
        # Calculate growth trend
        growth_count = sum(
            1 for i in range(1, len(allocated_values))
            if allocated_values[i] > allocated_values[i-1]
        )
        
        if growth_count > len(allocated_values) * 0.7:  # Growing 70%+ of the time
            leaks.append({
                "type": "continuous_growth",
                "start_mb": allocated_values[0],
                "end_mb": allocated_values[-1],
                "growth_mb": allocated_values[-1] - allocated_values[0],
                "recommendation": "Memory is continuously growing. Check for accumulating tensors or gradients."
            })
        
        return leaks
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self._torch_available:
            self._torch.cuda.reset_peak_memory_stats()
    
    def empty_cache(self) -> Dict[str, Any]:
        """Empty CUDA cache and return freed memory."""
        if not self._torch_available:
            return {}
        
        before = self.get_memory_summary()
        self._torch.cuda.empty_cache()
        after = self.get_memory_summary()
        
        return {
            "freed_mb": before["reserved_mb"] - after["reserved_mb"]
        }


# =============================================================================
# NVTX Markers for Custom Annotations
# =============================================================================

class NVTXMarker:
    """
    NVIDIA Tools Extension (NVTX) markers for custom profiling annotations.
    
    NVTX markers appear in Nsight Systems and other NVIDIA profiling tools,
    allowing you to correlate application events with GPU activity.
    
    Example:
        marker = NVTXMarker()
        with marker.range("inference"):
            model(input)
    """
    
    def __init__(self):
        self._nvtx_available = False
        
        try:
            import torch.cuda.nvtx as nvtx
            self._nvtx_available = True
            self._nvtx = nvtx
        except ImportError:
            try:
                import nvtx
                self._nvtx_available = True
                self._nvtx = nvtx
            except ImportError:
                logger.warning("NVTX not available")
    
    @contextmanager
    def range(self, name: str, color: str = "blue"):
        """Create an NVTX range marker."""
        if self._nvtx_available:
            self._nvtx.range_push(name)
            try:
                yield
            finally:
                self._nvtx.range_pop()
        else:
            yield
    
    def mark(self, name: str) -> None:
        """Create an instant NVTX marker."""
        if self._nvtx_available:
            self._nvtx.mark(name)


# =============================================================================
# Convenience Functions
# =============================================================================

def profile_inference(
    model_callable: Callable,
    input_data: Any,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, Any]:
    """
    Profile model inference with comprehensive metrics.
    
    Args:
        model_callable: Function that performs inference
        input_data: Input data for the model
        num_iterations: Number of timing iterations
        warmup_iterations: Warmup iterations
    
    Returns:
        Dictionary with timing and memory metrics
    """
    cuda_timer = CUDAEventTimer()
    memory_analyzer = GPUMemoryAnalyzer()
    
    latencies = []
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = model_callable(input_data)
    
    # Clear cache and reset stats
    memory_analyzer.empty_cache()
    memory_analyzer.reset_peak_stats()
    
    # Timed runs
    memory_analyzer.take_snapshot("before_inference")
    
    for i in range(num_iterations):
        cuda_timer.start()
        _ = model_callable(input_data)
        latencies.append(cuda_timer.stop())
    
    memory_analyzer.take_snapshot("after_inference")
    
    # Calculate statistics
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "num_iterations": num_iterations,
        "latency_mean_ms": sum(latencies) / n,
        "latency_std_ms": (sum((x - sum(latencies)/n)**2 for x in latencies) / n) ** 0.5,
        "latency_min_ms": sorted_latencies[0],
        "latency_max_ms": sorted_latencies[-1],
        "latency_p50_ms": sorted_latencies[n // 2],
        "latency_p90_ms": sorted_latencies[int(n * 0.9)],
        "latency_p99_ms": sorted_latencies[int(n * 0.99)],
        "throughput_qps": 1000 / (sum(latencies) / n),
        "memory": memory_analyzer.get_memory_summary()
    }
