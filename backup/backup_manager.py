"""
Backup Manager - Automated backup procedures.
"""

import os
import subprocess
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class BackupConfig:
    """Backup configuration."""
    name: str
    source_paths: List[str]
    destination: str
    storage_backend: StorageBackend = StorageBackend.LOCAL
    backup_type: BackupType = BackupType.FULL
    retention_days: int = 30
    compression: bool = True
    encryption_key: Optional[str] = None


@dataclass
class BackupResult:
    """Result of a backup operation."""
    backup_id: str
    config_name: str
    timestamp: datetime
    size_bytes: int
    duration_seconds: float
    success: bool
    files_backed_up: int = 0
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "config_name": self.config_name,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "files_backed_up": self.files_backed_up,
            "error_message": self.error_message,
            "checksum": self.checksum,
        }


class BackupManager:
    """
    Manages automated backup procedures.
    """
    
    def __init__(self, configs: List[BackupConfig] = None):
        self.configs = {c.name: c for c in (configs or [])}
        self.backup_history: List[BackupResult] = []
    
    def add_config(self, config: BackupConfig) -> None:
        """Add a backup configuration."""
        self.configs[config.name] = config
    
    def run_backup(self, config_name: str) -> BackupResult:
        """
        Execute backup for given configuration.
        """
        import time
        import uuid
        
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")
        
        config = self.configs[config_name]
        start_time = time.time()
        backup_id = f"{config_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        
        try:
            if config.storage_backend == StorageBackend.LOCAL:
                size, count, checksum = self._backup_local(config, backup_id)
            elif config.storage_backend == StorageBackend.S3:
                size, count, checksum = self._backup_s3(config, backup_id)
            else:
                raise NotImplementedError(f"Backend {config.storage_backend} not implemented")
            
            result = BackupResult(
                backup_id=backup_id,
                config_name=config_name,
                timestamp=datetime.now(),
                size_bytes=size,
                duration_seconds=time.time() - start_time,
                success=True,
                files_backed_up=count,
                checksum=checksum,
            )
        except Exception as e:
            result = BackupResult(
                backup_id=backup_id,
                config_name=config_name,
                timestamp=datetime.now(),
                size_bytes=0,
                duration_seconds=time.time() - start_time,
                success=False,
                error_message=str(e),
            )
        
        self.backup_history.append(result)
        return result
    
    def _backup_local(self, config: BackupConfig, backup_id: str):
        """Backup to local filesystem."""
        dest_path = Path(config.destination) / backup_id
        dest_path.mkdir(parents=True, exist_ok=True)
        
        total_size = 0
        file_count = 0
        hash_obj = hashlib.sha256()
        
        for source in config.source_paths:
            source_path = Path(source)
            if source_path.is_file():
                self._copy_file(source_path, dest_path, config.compression)
                total_size += source_path.stat().st_size
                file_count += 1
                hash_obj.update(source_path.read_bytes())
            elif source_path.is_dir():
                for file in source_path.rglob("*"):
                    if file.is_file():
                        rel_path = file.relative_to(source_path)
                        target = dest_path / source_path.name / rel_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        self._copy_file(file, target.parent, config.compression)
                        total_size += file.stat().st_size
                        file_count += 1
        
        # Write manifest
        manifest = {
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "files": file_count,
            "size": total_size,
        }
        (dest_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        return total_size, file_count, hash_obj.hexdigest()
    
    def _copy_file(self, source: Path, dest_dir: Path, compress: bool):
        """Copy file, optionally compressing."""
        import shutil
        import gzip
        
        dest_file = dest_dir / source.name
        
        if compress:
            dest_file = dest_dir / f"{source.name}.gz"
            with open(source, 'rb') as f_in:
                with gzip.open(dest_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(source, dest_file)
    
    def _backup_s3(self, config: BackupConfig, backup_id: str):
        """Backup to AWS S3."""
        # Use AWS CLI for S3 sync
        for source in config.source_paths:
            dest = f"{config.destination}/{backup_id}"
            cmd = ["aws", "s3", "sync", source, dest]
            subprocess.run(cmd, check=True)
        
        return 0, 0, None  # S3 handles sizing
    
    def list_backups(self, config_name: str = None) -> List[BackupResult]:
        """List backup history."""
        if config_name:
            return [b for b in self.backup_history if b.config_name == config_name]
        return self.backup_history
    
    def cleanup_old_backups(self, config_name: str) -> int:
        """Remove backups older than retention period."""
        if config_name not in self.configs:
            return 0
        
        config = self.configs[config_name]
        cutoff = datetime.now().timestamp() - (config.retention_days * 86400)
        removed = 0
        
        if config.storage_backend == StorageBackend.LOCAL:
            dest_path = Path(config.destination)
            for backup_dir in dest_path.iterdir():
                if backup_dir.is_dir() and backup_dir.name.startswith(config_name):
                    manifest_file = backup_dir / "manifest.json"
                    if manifest_file.exists():
                        manifest = json.loads(manifest_file.read_text())
                        backup_time = datetime.fromisoformat(manifest["timestamp"])
                        if backup_time.timestamp() < cutoff:
                            import shutil
                            shutil.rmtree(backup_dir)
                            removed += 1
        
        return removed
