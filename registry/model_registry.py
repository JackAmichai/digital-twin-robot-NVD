"""Model registry for ML model versioning and deployment."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import shutil


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    latency_ms: float = 0.0
    custom: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """Model version information."""
    
    version: str
    model_name: str
    stage: ModelStage = ModelStage.DEVELOPMENT
    artifact_path: str = ""
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    
    @property
    def full_name(self) -> str:
        """Get full model name with version."""
        return f"{self.model_name}:{self.version}"


class ModelRegistry:
    """Registry for ML model versioning and management."""
    
    def __init__(self, storage_path: Path | str = "./model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, dict[str, ModelVersion]] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load registry index from disk."""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self._models[model_name] = {}
                    for ver, info in versions.items():
                        self._models[model_name][ver] = ModelVersion(
                            version=ver,
                            model_name=model_name,
                            stage=ModelStage(info["stage"]),
                            artifact_path=info["artifact_path"],
                            description=info.get("description", ""),
                        )
    
    def _save_index(self) -> None:
        """Save registry index to disk."""
        data = {}
        for model_name, versions in self._models.items():
            data[model_name] = {}
            for ver, model in versions.items():
                data[model_name][ver] = {
                    "stage": model.stage.value,
                    "artifact_path": model.artifact_path,
                    "description": model.description,
                }
        
        with open(self.storage_path / "index.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def register(
        self,
        model_name: str,
        artifact_path: Path | str,
        version: str | None = None,
        metrics: ModelMetrics | None = None,
        parameters: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
    ) -> ModelVersion:
        """Register a new model version."""
        if model_name not in self._models:
            self._models[model_name] = {}
        
        # Auto-generate version if not provided
        if version is None:
            existing = len(self._models[model_name])
            version = f"v{existing + 1}"
        
        # Copy artifact to registry storage
        dest_path = self.storage_path / model_name / version
        dest_path.mkdir(parents=True, exist_ok=True)
        
        src = Path(artifact_path)
        if src.is_file():
            shutil.copy2(src, dest_path / src.name)
        elif src.is_dir():
            shutil.copytree(src, dest_path / "artifacts", dirs_exist_ok=True)
        
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            artifact_path=str(dest_path),
            metrics=metrics or ModelMetrics(),
            parameters=parameters or {},
            tags=tags or {},
            description=description,
        )
        
        self._models[model_name][version] = model_version
        self._save_index()
        
        return model_version
    
    def get_version(
        self,
        model_name: str,
        version: str | None = None,
        stage: ModelStage | None = None,
    ) -> ModelVersion | None:
        """Get specific model version or latest by stage."""
        if model_name not in self._models:
            return None
        
        versions = self._models[model_name]
        
        if version:
            return versions.get(version)
        
        if stage:
            for v in reversed(list(versions.values())):
                if v.stage == stage:
                    return v
        
        # Return latest
        if versions:
            return list(versions.values())[-1]
        
        return None
    
    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all versions of a model."""
        return list(self._models.get(model_name, {}).values())
    
    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
    ) -> bool:
        """Transition model to new stage."""
        model = self.get_version(model_name, version)
        if model:
            model.stage = stage
            self._save_index()
            return True
        return False
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a model version."""
        if model_name in self._models and version in self._models[model_name]:
            model = self._models[model_name][version]
            artifact_path = Path(model.artifact_path)
            if artifact_path.exists():
                shutil.rmtree(artifact_path)
            del self._models[model_name][version]
            self._save_index()
            return True
        return False


# Module-level convenience functions
_registry: ModelRegistry | None = None


def get_registry(storage_path: str = "./model_registry") -> ModelRegistry:
    """Get or create global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(storage_path)
    return _registry


def register_model(
    model_name: str,
    artifact_path: str,
    **kwargs: Any,
) -> ModelVersion:
    """Register model using global registry."""
    return get_registry().register(model_name, artifact_path, **kwargs)


def load_model(
    model_name: str,
    version: str | None = None,
    stage: ModelStage | None = None,
) -> ModelVersion | None:
    """Load model from global registry."""
    return get_registry().get_version(model_name, version, stage)
