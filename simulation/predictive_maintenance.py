#!/usr/bin/env python3
"""
Predictive Maintenance System for Robot Fleet

Analyzes simulation data and real-time telemetry to predict component
failures before they occur. Uses pattern recognition and ML models to:
- Track component wear over time
- Detect anomalies in sensor readings
- Predict remaining useful life (RUL)
- Generate maintenance alerts and schedules

Components Monitored:
- Motors (drive, steering, arm joints)
- Batteries (capacity degradation, cycle count)
- Wheels/Tracks (wear patterns, alignment)
- Bearings (vibration analysis)
- Sensors (drift, calibration)
- Gearboxes (efficiency, temperature)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Callable, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import threading
import json
import logging
import time
import hashlib

# Optional ML libraries
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available, using heuristic models")

# Optional ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32MultiArray
    from sensor_msgs.msg import JointState, BatteryState, Imu
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# Optional Redis for persistence
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class ComponentType(Enum):
    """Types of components that can be monitored."""
    MOTOR = "motor"
    BATTERY = "battery"
    WHEEL = "wheel"
    BEARING = "bearing"
    GEARBOX = "gearbox"
    SENSOR = "sensor"
    ENCODER = "encoder"
    CONTROLLER = "controller"


class AlertSeverity(Enum):
    """Severity levels for maintenance alerts."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Attention needed soon
    CRITICAL = "critical"   # Immediate attention required
    EMERGENCY = "emergency" # System shutdown recommended


class MaintenanceAction(Enum):
    """Recommended maintenance actions."""
    INSPECT = "inspect"
    LUBRICATE = "lubricate"
    CALIBRATE = "calibrate"
    REPLACE = "replace"
    REPAIR = "repair"
    CLEAN = "clean"
    RECHARGE = "recharge"


@dataclass
class ComponentSpec:
    """Specification for a monitored component."""
    component_id: str
    component_type: ComponentType
    robot_id: str
    
    # Design specs
    rated_hours: float = 10000.0          # Expected lifetime in hours
    rated_cycles: int = 100000            # Expected cycle count
    
    # Operational limits
    max_temperature_c: float = 80.0
    max_current_a: float = 10.0
    max_vibration_g: float = 2.0
    
    # Warning thresholds (% of limit)
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['component_type'] = self.component_type.value
        return result


@dataclass
class ComponentHealth:
    """Current health status of a component."""
    component_id: str
    health_score: float = 1.0           # 0 = failed, 1 = perfect
    remaining_life_hours: float = 0.0   # Estimated RUL
    remaining_life_pct: float = 100.0   # % of expected lifetime
    
    # Wear metrics
    operating_hours: float = 0.0
    cycle_count: int = 0
    
    # Current readings
    temperature_c: float = 25.0
    current_a: float = 0.0
    vibration_g: float = 0.0
    efficiency_pct: float = 100.0
    
    # Trends
    health_trend: str = "stable"        # improving, stable, degrading
    degradation_rate: float = 0.0       # % per hour
    
    # Timestamps
    last_maintenance: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'component_id': self.component_id,
            'health_score': self.health_score,
            'remaining_life_hours': self.remaining_life_hours,
            'remaining_life_pct': self.remaining_life_pct,
            'operating_hours': self.operating_hours,
            'cycle_count': self.cycle_count,
            'temperature_c': self.temperature_c,
            'current_a': self.current_a,
            'vibration_g': self.vibration_g,
            'efficiency_pct': self.efficiency_pct,
            'health_trend': self.health_trend,
            'degradation_rate': self.degradation_rate,
            'last_maintenance': self.last_maintenance.isoformat() if self.last_maintenance else None,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class MaintenanceAlert:
    """Maintenance alert/recommendation."""
    alert_id: str
    component_id: str
    robot_id: str
    severity: AlertSeverity
    action: MaintenanceAction
    message: str
    details: str
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> dict:
        return {
            'alert_id': self.alert_id,
            'component_id': self.component_id,
            'robot_id': self.robot_id,
            'severity': self.severity.value,
            'action': self.action.value,
            'message': self.message,
            'details': self.details,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


@dataclass
class TelemetryReading:
    """Single telemetry reading from a component."""
    component_id: str
    timestamp: datetime
    values: Dict[str, float]  # metric_name -> value
    
    def to_dict(self) -> dict:
        return {
            'component_id': self.component_id,
            'timestamp': self.timestamp.isoformat(),
            'values': self.values
        }


# =============================================================================
# Wear Models
# =============================================================================

class WearModel:
    """Base class for component wear models."""
    
    def __init__(self, component_spec: ComponentSpec):
        self.spec = component_spec
        
    def calculate_health(self, telemetry: List[TelemetryReading]) -> float:
        """Calculate health score from telemetry. Override in subclasses."""
        raise NotImplementedError
        
    def predict_rul(self, health_history: List[float]) -> float:
        """Predict remaining useful life in hours."""
        raise NotImplementedError


class MotorWearModel(WearModel):
    """
    Wear model for electric motors.
    
    Factors considered:
    - Operating hours
    - Temperature cycles
    - Current draw patterns
    - Vibration levels
    - Bearing wear
    """
    
    def __init__(self, component_spec: ComponentSpec):
        super().__init__(component_spec)
        
        # Motor-specific parameters
        self.temp_weight = 0.3
        self.current_weight = 0.2
        self.vibration_weight = 0.3
        self.hours_weight = 0.2
        
        # Degradation factors
        self.high_temp_factor = 2.0      # Doubles wear above 70% of max temp
        self.overload_factor = 1.5       # 50% more wear when overloaded
        
    def calculate_health(self, telemetry: List[TelemetryReading]) -> float:
        if not telemetry:
            return 1.0
            
        latest = telemetry[-1]
        values = latest.values
        
        # Temperature score
        temp = values.get('temperature', 25.0)
        temp_ratio = temp / self.spec.max_temperature_c
        temp_score = 1.0 - min(1.0, temp_ratio * self.temp_weight)
        
        # Current score
        current = values.get('current', 0.0)
        current_ratio = current / self.spec.max_current_a
        current_score = 1.0 - min(1.0, current_ratio * self.current_weight)
        
        # Vibration score
        vibration = values.get('vibration', 0.0)
        vib_ratio = vibration / self.spec.max_vibration_g
        vib_score = 1.0 - min(1.0, vib_ratio * self.vibration_weight)
        
        # Hours score
        hours = values.get('operating_hours', 0.0)
        hours_ratio = hours / self.spec.rated_hours
        hours_score = 1.0 - min(1.0, hours_ratio * self.hours_weight)
        
        # Weighted combination
        health = (
            temp_score * self.temp_weight +
            current_score * self.current_weight +
            vib_score * self.vibration_weight +
            hours_score * self.hours_weight
        )
        
        return max(0.0, min(1.0, health))
        
    def predict_rul(self, health_history: List[float]) -> float:
        if len(health_history) < 2:
            return self.spec.rated_hours
            
        # Calculate degradation rate
        recent = health_history[-10:]  # Last 10 readings
        if len(recent) < 2:
            return self.spec.rated_hours
            
        rate = (recent[0] - recent[-1]) / len(recent)
        
        if rate <= 0:
            return self.spec.rated_hours  # No degradation
            
        current_health = recent[-1]
        
        # Time to reach 20% health (failure threshold)
        failure_threshold = 0.2
        hours_to_failure = (current_health - failure_threshold) / rate
        
        return max(0, hours_to_failure)


class BatteryWearModel(WearModel):
    """
    Wear model for robot batteries.
    
    Factors considered:
    - Charge cycles
    - Depth of discharge
    - Temperature during charging
    - Calendar aging
    - Fast charge usage
    """
    
    def __init__(self, component_spec: ComponentSpec):
        super().__init__(component_spec)
        
        # Battery-specific parameters
        self.cycle_weight = 0.4
        self.temp_weight = 0.2
        self.capacity_weight = 0.3
        self.age_weight = 0.1
        
    def calculate_health(self, telemetry: List[TelemetryReading]) -> float:
        if not telemetry:
            return 1.0
            
        latest = telemetry[-1]
        values = latest.values
        
        # Cycle degradation
        cycles = values.get('cycle_count', 0)
        cycle_ratio = cycles / self.spec.rated_cycles
        cycle_score = 1.0 - min(1.0, cycle_ratio)
        
        # Capacity retention
        capacity = values.get('capacity_pct', 100.0)
        capacity_score = capacity / 100.0
        
        # Temperature impact
        temp = values.get('temperature', 25.0)
        # Optimal range is 20-30°C
        if 20 <= temp <= 30:
            temp_score = 1.0
        else:
            temp_deviation = min(abs(temp - 20), abs(temp - 30))
            temp_score = 1.0 - (temp_deviation / 50.0)
            
        # Age factor (assume 5 year lifetime)
        age_months = values.get('age_months', 0)
        age_score = 1.0 - min(1.0, age_months / 60.0)
        
        health = (
            cycle_score * self.cycle_weight +
            capacity_score * self.capacity_weight +
            temp_score * self.temp_weight +
            age_score * self.age_weight
        )
        
        return max(0.0, min(1.0, health))
        
    def predict_rul(self, health_history: List[float]) -> float:
        if len(health_history) < 5:
            return 1000.0  # Default hours
            
        # Simple linear projection
        recent = health_history[-5:]
        rate = (recent[0] - recent[-1]) / len(recent)
        
        if rate <= 0:
            return 5000.0
            
        current = recent[-1]
        hours_to_80pct = (current - 0.8) / rate if current > 0.8 else 0
        
        return max(0, hours_to_80pct * 100)  # Convert to hours estimate


class WheelWearModel(WearModel):
    """
    Wear model for robot wheels/tracks.
    
    Factors considered:
    - Distance traveled
    - Surface roughness
    - Load patterns
    - Alignment
    - Tread depth
    """
    
    def calculate_health(self, telemetry: List[TelemetryReading]) -> float:
        if not telemetry:
            return 1.0
            
        latest = telemetry[-1]
        values = latest.values
        
        # Tread depth (mm)
        tread = values.get('tread_depth_mm', 10.0)
        min_tread = 2.0
        max_tread = 10.0
        tread_score = (tread - min_tread) / (max_tread - min_tread)
        
        # Distance factor
        distance_km = values.get('distance_km', 0.0)
        rated_distance = 50000  # 50,000 km rated
        distance_score = 1.0 - min(1.0, distance_km / rated_distance)
        
        # Alignment factor
        alignment_deg = values.get('alignment_error_deg', 0.0)
        alignment_score = 1.0 - min(1.0, alignment_deg / 5.0)
        
        health = tread_score * 0.5 + distance_score * 0.3 + alignment_score * 0.2
        
        return max(0.0, min(1.0, health))
        
    def predict_rul(self, health_history: List[float]) -> float:
        if len(health_history) < 3:
            return 500.0
            
        recent = health_history[-5:]
        rate = (recent[0] - recent[-1]) / len(recent)
        
        if rate <= 0:
            return 2000.0
            
        current = recent[-1]
        # Wheels fail at 30% health
        hours_to_fail = (current - 0.3) / rate if current > 0.3 else 0
        
        return max(0, hours_to_fail * 50)


class AnomalyDetector:
    """
    Detects anomalies in telemetry using statistical methods.
    Uses Isolation Forest when sklearn is available.
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.history: deque = deque(maxlen=1000)
        self.model = None
        self.scaler = None
        
        if HAS_SKLEARN:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.is_fitted = False
        else:
            self.baseline_stats: Dict[str, Tuple[float, float]] = {}  # metric -> (mean, std)
            
    def update(self, reading: TelemetryReading):
        """Update the detector with new data."""
        values = list(reading.values.values())
        self.history.append(values)
        
        # Refit periodically
        if len(self.history) >= 100 and len(self.history) % 100 == 0:
            self._fit()
            
    def _fit(self):
        """Fit the anomaly detection model."""
        if len(self.history) < 50:
            return
            
        data = np.array(list(self.history))
        
        if HAS_SKLEARN:
            self.scaler.fit(data)
            scaled_data = self.scaler.transform(data)
            self.model.fit(scaled_data)
            self.is_fitted = True
        else:
            # Simple stats baseline
            for i in range(data.shape[1]):
                self.baseline_stats[i] = (np.mean(data[:, i]), np.std(data[:, i]))
                
    def is_anomaly(self, reading: TelemetryReading) -> Tuple[bool, float]:
        """
        Check if reading is anomalous.
        
        Returns:
            (is_anomaly, anomaly_score)
        """
        values = np.array(list(reading.values.values())).reshape(1, -1)
        
        if HAS_SKLEARN and self.is_fitted:
            scaled = self.scaler.transform(values)
            prediction = self.model.predict(scaled)[0]
            score = -self.model.score_samples(scaled)[0]  # Higher = more anomalous
            return (prediction == -1, score)
        else:
            # Z-score based detection
            if not self.baseline_stats:
                return (False, 0.0)
                
            z_scores = []
            for i, val in enumerate(values[0]):
                if i in self.baseline_stats:
                    mean, std = self.baseline_stats[i]
                    if std > 0:
                        z = abs(val - mean) / std
                        z_scores.append(z)
                        
            if not z_scores:
                return (False, 0.0)
                
            max_z = max(z_scores)
            return (max_z > 3.0, max_z / 5.0)  # Normalize score


# =============================================================================
# Predictive Maintenance Engine
# =============================================================================

class PredictiveMaintenanceEngine:
    """
    Core engine for predictive maintenance.
    
    Manages component monitoring, health tracking, and alert generation.
    
    Example Usage:
        >>> engine = PredictiveMaintenanceEngine()
        >>> 
        >>> # Register components
        >>> engine.register_component(ComponentSpec(
        ...     component_id="motor_left",
        ...     component_type=ComponentType.MOTOR,
        ...     robot_id="robot_1"
        ... ))
        >>> 
        >>> # Feed telemetry
        >>> engine.update_telemetry(TelemetryReading(
        ...     component_id="motor_left",
        ...     timestamp=datetime.now(),
        ...     values={'temperature': 45.0, 'current': 5.0}
        ... ))
        >>> 
        >>> # Get health status
        >>> health = engine.get_component_health("motor_left")
        >>> print(f"Health: {health.health_score:.1%}")
        >>> 
        >>> # Get alerts
        >>> alerts = engine.get_active_alerts("robot_1")
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        use_redis: bool = True
    ):
        # Component registry
        self._components: Dict[str, ComponentSpec] = {}
        self._health: Dict[str, ComponentHealth] = {}
        self._wear_models: Dict[str, WearModel] = {}
        self._anomaly_detectors: Dict[str, AnomalyDetector] = {}
        
        # Telemetry history
        self._telemetry: Dict[str, deque] = {}  # component_id -> readings
        self._health_history: Dict[str, deque] = {}  # component_id -> health scores
        
        # Alerts
        self._alerts: Dict[str, MaintenanceAlert] = {}  # alert_id -> alert
        self._alert_callbacks: List[Callable[[MaintenanceAlert], None]] = []
        
        # Redis persistence
        self._redis = None
        if use_redis and HAS_REDIS:
            try:
                self._redis = redis.Redis(
                    host=redis_host, 
                    port=redis_port,
                    decode_responses=True
                )
                self._redis.ping()
                logger.info("Redis persistence enabled")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("Predictive Maintenance Engine initialized")
        
    def register_component(self, spec: ComponentSpec):
        """Register a component for monitoring."""
        with self._lock:
            self._components[spec.component_id] = spec
            
            # Initialize health
            self._health[spec.component_id] = ComponentHealth(
                component_id=spec.component_id,
                remaining_life_hours=spec.rated_hours,
                remaining_life_pct=100.0
            )
            
            # Create wear model
            wear_models = {
                ComponentType.MOTOR: MotorWearModel,
                ComponentType.BATTERY: BatteryWearModel,
                ComponentType.WHEEL: WheelWearModel,
            }
            model_class = wear_models.get(spec.component_type, MotorWearModel)
            self._wear_models[spec.component_id] = model_class(spec)
            
            # Create anomaly detector
            self._anomaly_detectors[spec.component_id] = AnomalyDetector()
            
            # Initialize histories
            self._telemetry[spec.component_id] = deque(maxlen=1000)
            self._health_history[spec.component_id] = deque(maxlen=500)
            
            logger.info(f"Registered component: {spec.component_id} ({spec.component_type.value})")
            
    def update_telemetry(self, reading: TelemetryReading):
        """Process a new telemetry reading."""
        with self._lock:
            if reading.component_id not in self._components:
                logger.warning(f"Unknown component: {reading.component_id}")
                return
                
            # Store reading
            self._telemetry[reading.component_id].append(reading)
            
            # Update anomaly detector
            detector = self._anomaly_detectors[reading.component_id]
            detector.update(reading)
            
            # Check for anomalies
            is_anomaly, score = detector.is_anomaly(reading)
            if is_anomaly:
                self._generate_anomaly_alert(reading.component_id, score)
                
            # Update health
            self._update_health(reading.component_id)
            
            # Persist to Redis
            if self._redis:
                self._persist_reading(reading)
                
    def _update_health(self, component_id: str):
        """Recalculate health for a component."""
        spec = self._components[component_id]
        model = self._wear_models[component_id]
        telemetry = list(self._telemetry[component_id])
        
        # Calculate current health
        health_score = model.calculate_health(telemetry)
        
        # Update health history
        self._health_history[component_id].append(health_score)
        
        # Predict RUL
        history = list(self._health_history[component_id])
        rul_hours = model.predict_rul(history)
        
        # Calculate trend
        trend = self._calculate_trend(history)
        degradation_rate = self._calculate_degradation_rate(history)
        
        # Update health record
        health = self._health[component_id]
        health.health_score = health_score
        health.remaining_life_hours = rul_hours
        health.remaining_life_pct = (rul_hours / spec.rated_hours) * 100 if spec.rated_hours > 0 else 0
        health.health_trend = trend
        health.degradation_rate = degradation_rate
        health.last_updated = datetime.now()
        
        # Update metrics from telemetry
        if telemetry:
            latest = telemetry[-1]
            health.temperature_c = latest.values.get('temperature', health.temperature_c)
            health.current_a = latest.values.get('current', health.current_a)
            health.vibration_g = latest.values.get('vibration', health.vibration_g)
            health.operating_hours = latest.values.get('operating_hours', health.operating_hours)
            health.cycle_count = int(latest.values.get('cycle_count', health.cycle_count))
            
        # Check thresholds and generate alerts
        self._check_thresholds(component_id, health)
        
    def _calculate_trend(self, history: List[float]) -> str:
        """Calculate health trend."""
        if len(history) < 5:
            return "stable"
            
        recent = history[-5:]
        change = recent[-1] - recent[0]
        
        if change > 0.02:
            return "improving"
        elif change < -0.02:
            return "degrading"
        return "stable"
        
    def _calculate_degradation_rate(self, history: List[float]) -> float:
        """Calculate degradation rate (% per hour)."""
        if len(history) < 10:
            return 0.0
            
        recent = history[-10:]
        rate = (recent[0] - recent[-1]) / len(recent)
        return max(0.0, rate * 100)  # Convert to percentage
        
    def _check_thresholds(self, component_id: str, health: ComponentHealth):
        """Check health against thresholds and generate alerts."""
        spec = self._components[component_id]
        
        # Critical health
        if health.health_score < 1 - spec.critical_threshold:
            self._generate_alert(
                component_id=component_id,
                severity=AlertSeverity.CRITICAL,
                action=MaintenanceAction.REPLACE,
                message=f"Critical health level: {health.health_score:.1%}",
                details=f"Component {component_id} has degraded to critical levels. "
                        f"Estimated {health.remaining_life_hours:.0f} hours remaining."
            )
        # Warning health
        elif health.health_score < 1 - spec.warning_threshold:
            self._generate_alert(
                component_id=component_id,
                severity=AlertSeverity.WARNING,
                action=MaintenanceAction.INSPECT,
                message=f"Health declining: {health.health_score:.1%}",
                details=f"Component {component_id} showing wear. "
                        f"Schedule inspection within {health.remaining_life_hours:.0f} hours."
            )
            
        # Temperature check
        if health.temperature_c > spec.max_temperature_c * spec.critical_threshold:
            self._generate_alert(
                component_id=component_id,
                severity=AlertSeverity.CRITICAL,
                action=MaintenanceAction.INSPECT,
                message=f"High temperature: {health.temperature_c:.1f}°C",
                details=f"Temperature exceeds safe operating range "
                        f"(max: {spec.max_temperature_c}°C)"
            )
            
        # Vibration check
        if health.vibration_g > spec.max_vibration_g * spec.warning_threshold:
            self._generate_alert(
                component_id=component_id,
                severity=AlertSeverity.WARNING,
                action=MaintenanceAction.LUBRICATE,
                message=f"High vibration: {health.vibration_g:.2f}g",
                details="Elevated vibration may indicate bearing wear or misalignment"
            )
            
    def _generate_alert(
        self,
        component_id: str,
        severity: AlertSeverity,
        action: MaintenanceAction,
        message: str,
        details: str
    ):
        """Generate a maintenance alert."""
        spec = self._components[component_id]
        
        # Create unique ID
        alert_id = hashlib.md5(
            f"{component_id}:{severity.value}:{message}".encode()
        ).hexdigest()[:12]
        
        # Check for duplicate
        if alert_id in self._alerts and not self._alerts[alert_id].resolved:
            return  # Don't duplicate
            
        alert = MaintenanceAlert(
            alert_id=alert_id,
            component_id=component_id,
            robot_id=spec.robot_id,
            severity=severity,
            action=action,
            message=message,
            details=details
        )
        
        self._alerts[alert_id] = alert
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
        # Persist
        if self._redis:
            self._redis.hset(
                f"maintenance:alerts:{spec.robot_id}",
                alert_id,
                json.dumps(alert.to_dict())
            )
            
        logger.warning(f"Alert generated: [{severity.value}] {message}")
        
    def _generate_anomaly_alert(self, component_id: str, score: float):
        """Generate alert for anomalous reading."""
        self._generate_alert(
            component_id=component_id,
            severity=AlertSeverity.WARNING,
            action=MaintenanceAction.INSPECT,
            message=f"Anomalous sensor reading (score: {score:.2f})",
            details="Telemetry values deviate significantly from normal patterns. "
                    "Possible sensor malfunction or unusual operating conditions."
        )
        
    def _persist_reading(self, reading: TelemetryReading):
        """Persist telemetry to Redis."""
        key = f"telemetry:{reading.component_id}"
        self._redis.lpush(key, json.dumps(reading.to_dict()))
        self._redis.ltrim(key, 0, 999)  # Keep last 1000
        
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get current health of a component."""
        return self._health.get(component_id)
        
    def get_robot_health_summary(self, robot_id: str) -> Dict[str, Any]:
        """Get health summary for all components of a robot."""
        summary = {
            'robot_id': robot_id,
            'overall_health': 1.0,
            'components': {},
            'alerts_count': 0,
            'critical_alerts': 0
        }
        
        healths = []
        
        for comp_id, spec in self._components.items():
            if spec.robot_id == robot_id:
                health = self._health.get(comp_id)
                if health:
                    summary['components'][comp_id] = health.to_dict()
                    healths.append(health.health_score)
                    
        # Overall health is minimum of all components
        if healths:
            summary['overall_health'] = min(healths)
            
        # Count alerts
        for alert in self._alerts.values():
            if alert.robot_id == robot_id and not alert.resolved:
                summary['alerts_count'] += 1
                if alert.severity == AlertSeverity.CRITICAL:
                    summary['critical_alerts'] += 1
                    
        return summary
        
    def get_active_alerts(
        self,
        robot_id: Optional[str] = None,
        min_severity: AlertSeverity = AlertSeverity.INFO
    ) -> List[MaintenanceAlert]:
        """Get active (unresolved) alerts."""
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
            AlertSeverity.EMERGENCY: 3
        }
        min_level = severity_order[min_severity]
        
        alerts = []
        for alert in self._alerts.values():
            if alert.resolved:
                continue
            if robot_id and alert.robot_id != robot_id:
                continue
            if severity_order[alert.severity] < min_level:
                continue
            alerts.append(alert)
            
        # Sort by severity (highest first)
        alerts.sort(key=lambda a: severity_order[a.severity], reverse=True)
        return alerts
        
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            logger.info(f"Alert resolved: {alert_id}")
            
    def record_maintenance(
        self,
        component_id: str,
        action: MaintenanceAction,
        notes: str = ""
    ):
        """Record that maintenance was performed."""
        if component_id in self._health:
            self._health[component_id].last_maintenance = datetime.now()
            
        # Resolve related alerts
        for alert in self._alerts.values():
            if alert.component_id == component_id and not alert.resolved:
                alert.resolved = True
                
        logger.info(f"Maintenance recorded: {component_id} - {action.value}")
        
    def on_alert(self, callback: Callable[[MaintenanceAlert], None]):
        """Register callback for new alerts."""
        self._alert_callbacks.append(callback)
        
    def get_maintenance_schedule(
        self,
        robot_id: str,
        horizon_hours: float = 168  # 1 week
    ) -> List[Dict[str, Any]]:
        """
        Generate maintenance schedule based on predicted failures.
        
        Returns recommended maintenance actions sorted by urgency.
        """
        schedule = []
        
        for comp_id, spec in self._components.items():
            if spec.robot_id != robot_id:
                continue
                
            health = self._health.get(comp_id)
            if not health:
                continue
                
            # Components needing attention within horizon
            if health.remaining_life_hours < horizon_hours:
                urgency = 1.0 - (health.remaining_life_hours / horizon_hours)
                
                schedule.append({
                    'component_id': comp_id,
                    'component_type': spec.component_type.value,
                    'urgency': urgency,
                    'remaining_hours': health.remaining_life_hours,
                    'health_score': health.health_score,
                    'recommended_action': self._recommend_action(spec, health).value,
                    'estimated_downtime_hours': self._estimate_downtime(spec),
                })
                
        # Sort by urgency
        schedule.sort(key=lambda x: x['urgency'], reverse=True)
        return schedule
        
    def _recommend_action(
        self,
        spec: ComponentSpec,
        health: ComponentHealth
    ) -> MaintenanceAction:
        """Recommend maintenance action based on component state."""
        if health.health_score < 0.3:
            return MaintenanceAction.REPLACE
        elif health.health_score < 0.5:
            if spec.component_type == ComponentType.BEARING:
                return MaintenanceAction.LUBRICATE
            elif spec.component_type == ComponentType.SENSOR:
                return MaintenanceAction.CALIBRATE
            else:
                return MaintenanceAction.REPAIR
        elif health.vibration_g > spec.max_vibration_g * 0.5:
            return MaintenanceAction.LUBRICATE
        else:
            return MaintenanceAction.INSPECT
            
    def _estimate_downtime(self, spec: ComponentSpec) -> float:
        """Estimate downtime for maintenance (hours)."""
        downtimes = {
            ComponentType.MOTOR: 4.0,
            ComponentType.BATTERY: 2.0,
            ComponentType.WHEEL: 1.0,
            ComponentType.BEARING: 3.0,
            ComponentType.GEARBOX: 6.0,
            ComponentType.SENSOR: 0.5,
            ComponentType.ENCODER: 1.0,
            ComponentType.CONTROLLER: 2.0,
        }
        return downtimes.get(spec.component_type, 2.0)


# =============================================================================
# ROS 2 Integration
# =============================================================================

if HAS_ROS:
    class MaintenanceNode(Node):
        """ROS 2 node for predictive maintenance integration."""
        
        def __init__(self, engine: PredictiveMaintenanceEngine, robot_id: str):
            super().__init__(f'predictive_maintenance_{robot_id}')
            
            self.engine = engine
            self.robot_id = robot_id
            
            # Subscribers
            self.joint_sub = self.create_subscription(
                JointState,
                f'/{robot_id}/joint_states',
                self._joint_callback,
                10
            )
            
            self.battery_sub = self.create_subscription(
                BatteryState,
                f'/{robot_id}/battery_state',
                self._battery_callback,
                10
            )
            
            self.imu_sub = self.create_subscription(
                Imu,
                f'/{robot_id}/imu',
                self._imu_callback,
                10
            )
            
            # Publishers
            self.health_pub = self.create_publisher(
                String,
                f'/{robot_id}/maintenance/health',
                10
            )
            
            self.alert_pub = self.create_publisher(
                String,
                f'/{robot_id}/maintenance/alerts',
                10
            )
            
            # Timer for publishing status
            self.create_timer(5.0, self._publish_status)
            
            # Register alert callback
            engine.on_alert(self._on_alert)
            
            self.get_logger().info(f"Maintenance node started for {robot_id}")
            
        def _joint_callback(self, msg: JointState):
            """Process joint state for motor health."""
            for i, name in enumerate(msg.name):
                component_id = f"{self.robot_id}_{name}"
                
                values = {
                    'position': msg.position[i] if i < len(msg.position) else 0.0,
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0,
                }
                
                reading = TelemetryReading(
                    component_id=component_id,
                    timestamp=datetime.now(),
                    values=values
                )
                
                self.engine.update_telemetry(reading)
                
        def _battery_callback(self, msg: BatteryState):
            """Process battery state."""
            component_id = f"{self.robot_id}_battery"
            
            values = {
                'voltage': msg.voltage,
                'current': msg.current,
                'temperature': msg.temperature if msg.temperature > 0 else 25.0,
                'percentage': msg.percentage * 100,
                'capacity_pct': (msg.capacity / msg.design_capacity * 100) 
                               if msg.design_capacity > 0 else 100.0,
            }
            
            reading = TelemetryReading(
                component_id=component_id,
                timestamp=datetime.now(),
                values=values
            )
            
            self.engine.update_telemetry(reading)
            
        def _imu_callback(self, msg: Imu):
            """Process IMU for vibration analysis."""
            # Calculate vibration magnitude
            accel = msg.linear_acceleration
            vibration = np.sqrt(accel.x**2 + accel.y**2 + accel.z**2) - 9.81
            vibration = abs(vibration) / 9.81  # Convert to g's
            
            # This could indicate bearing or motor issues
            component_id = f"{self.robot_id}_chassis"
            
            values = {
                'vibration': vibration,
                'angular_x': msg.angular_velocity.x,
                'angular_y': msg.angular_velocity.y,
                'angular_z': msg.angular_velocity.z,
            }
            
            reading = TelemetryReading(
                component_id=component_id,
                timestamp=datetime.now(),
                values=values
            )
            
            self.engine.update_telemetry(reading)
            
        def _publish_status(self):
            """Publish health status."""
            summary = self.engine.get_robot_health_summary(self.robot_id)
            
            msg = String()
            msg.data = json.dumps(summary)
            self.health_pub.publish(msg)
            
        def _on_alert(self, alert: MaintenanceAlert):
            """Handle new alert."""
            if alert.robot_id != self.robot_id:
                return
                
            msg = String()
            msg.data = json.dumps(alert.to_dict())
            self.alert_pub.publish(msg)


# =============================================================================
# Demo
# =============================================================================

def demo_predictive_maintenance():
    """Demonstrate predictive maintenance system."""
    print("=== Predictive Maintenance Demo ===\n")
    
    # Create engine
    engine = PredictiveMaintenanceEngine(use_redis=False)
    
    # Alert callback
    def on_alert(alert: MaintenanceAlert):
        print(f"  🚨 [{alert.severity.value.upper()}] {alert.message}")
    
    engine.on_alert(on_alert)
    
    # Register components for robot_1
    components = [
        ComponentSpec(
            component_id="robot_1_motor_left",
            component_type=ComponentType.MOTOR,
            robot_id="robot_1",
            rated_hours=10000,
            max_temperature_c=80.0
        ),
        ComponentSpec(
            component_id="robot_1_motor_right",
            component_type=ComponentType.MOTOR,
            robot_id="robot_1",
            rated_hours=10000,
            max_temperature_c=80.0
        ),
        ComponentSpec(
            component_id="robot_1_battery",
            component_type=ComponentType.BATTERY,
            robot_id="robot_1",
            rated_cycles=500
        ),
        ComponentSpec(
            component_id="robot_1_wheel_left",
            component_type=ComponentType.WHEEL,
            robot_id="robot_1"
        ),
    ]
    
    for spec in components:
        engine.register_component(spec)
        
    print("Registered components:")
    for spec in components:
        print(f"  - {spec.component_id} ({spec.component_type.value})")
        
    print("\n--- Simulating telemetry stream ---\n")
    
    # Simulate degrading motor
    for hour in range(100):
        # Left motor - degrading
        engine.update_telemetry(TelemetryReading(
            component_id="robot_1_motor_left",
            timestamp=datetime.now(),
            values={
                'temperature': 35 + hour * 0.4,  # Rising temperature
                'current': 5.0 + hour * 0.02,    # Increasing current draw
                'vibration': 0.2 + hour * 0.01,  # Increasing vibration
                'operating_hours': hour * 100
            }
        ))
        
        # Right motor - healthy
        engine.update_telemetry(TelemetryReading(
            component_id="robot_1_motor_right",
            timestamp=datetime.now(),
            values={
                'temperature': 35 + np.random.normal(0, 2),
                'current': 5.0 + np.random.normal(0, 0.2),
                'vibration': 0.2 + np.random.normal(0, 0.05),
                'operating_hours': hour * 100
            }
        ))
        
        # Battery - slow degradation
        engine.update_telemetry(TelemetryReading(
            component_id="robot_1_battery",
            timestamp=datetime.now(),
            values={
                'temperature': 25,
                'capacity_pct': 100 - hour * 0.1,
                'cycle_count': hour * 5
            }
        ))
        
    print("\n--- Component Health Summary ---\n")
    
    summary = engine.get_robot_health_summary("robot_1")
    print(f"Robot: {summary['robot_id']}")
    print(f"Overall Health: {summary['overall_health']:.1%}")
    print(f"Active Alerts: {summary['alerts_count']} ({summary['critical_alerts']} critical)")
    
    print("\nComponent Details:")
    for comp_id, health in summary['components'].items():
        print(f"\n  {comp_id}:")
        print(f"    Health Score: {health['health_score']:.1%}")
        print(f"    Remaining Life: {health['remaining_life_hours']:.0f} hours")
        print(f"    Trend: {health['health_trend']}")
        print(f"    Temperature: {health['temperature_c']:.1f}°C")
        
    print("\n--- Maintenance Schedule ---\n")
    
    schedule = engine.get_maintenance_schedule("robot_1", horizon_hours=500)
    
    if schedule:
        print("Upcoming maintenance (next 500 hours):")
        for item in schedule:
            print(f"\n  {item['component_id']}:")
            print(f"    Urgency: {item['urgency']:.1%}")
            print(f"    Remaining: {item['remaining_hours']:.0f} hours")
            print(f"    Action: {item['recommended_action']}")
            print(f"    Downtime: {item['estimated_downtime_hours']:.1f} hours")
    else:
        print("No maintenance needed within the next 500 hours.")
        
    print("\n--- Active Alerts ---\n")
    
    alerts = engine.get_active_alerts("robot_1")
    if alerts:
        for alert in alerts:
            print(f"  [{alert.severity.value.upper()}] {alert.message}")
            print(f"    Component: {alert.component_id}")
            print(f"    Action: {alert.action.value}")
            print(f"    Details: {alert.details[:80]}...")
            print()
    else:
        print("No active alerts.")


if __name__ == "__main__":
    demo_predictive_maintenance()
