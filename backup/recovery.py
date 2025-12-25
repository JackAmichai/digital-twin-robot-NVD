"""
Recovery Manager - Disaster recovery procedures.
"""

import json
import gzip
import shutil
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class RecoveryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RecoveryResult:
    """Result of recovery operation."""
    backup_id: str
    target_path: str
    status: RecoveryStatus
    files_restored: int
    duration_seconds: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "target_path": self.target_path,
            "status": self.status.value,
            "files_restored": self.files_restored,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


class RecoveryManager:
    """
    Manages disaster recovery procedures.
    """
    
    def __init__(self, backup_root: str):
        self.backup_root = Path(backup_root)
    
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backups for recovery."""
        backups = []
        
        for backup_dir in self.backup_root.iterdir():
            if backup_dir.is_dir():
                manifest_file = backup_dir / "manifest.json"
                if manifest_file.exists():
                    manifest = json.loads(manifest_file.read_text())
                    manifest["path"] = str(backup_dir)
                    backups.append(manifest)
        
        # Sort by timestamp descending
        backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return backups
    
    def restore(
        self,
        backup_id: str,
        target_path: str,
        overwrite: bool = False
    ) -> RecoveryResult:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of backup to restore
            target_path: Where to restore files
            overwrite: Whether to overwrite existing files
        """
        import time
        
        start_time = time.time()
        backup_path = self.backup_root / backup_id
        target = Path(target_path)
        
        if not backup_path.exists():
            return RecoveryResult(
                backup_id=backup_id,
                target_path=target_path,
                status=RecoveryStatus.FAILED,
                files_restored=0,
                duration_seconds=time.time() - start_time,
                error_message=f"Backup not found: {backup_id}",
            )
        
        try:
            target.mkdir(parents=True, exist_ok=True)
            files_restored = self._restore_files(backup_path, target, overwrite)
            
            return RecoveryResult(
                backup_id=backup_id,
                target_path=target_path,
                status=RecoveryStatus.COMPLETED,
                files_restored=files_restored,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return RecoveryResult(
                backup_id=backup_id,
                target_path=target_path,
                status=RecoveryStatus.FAILED,
                files_restored=0,
                duration_seconds=time.time() - start_time,
                error_message=str(e),
            )
    
    def _restore_files(self, source: Path, target: Path, overwrite: bool) -> int:
        """Restore files from backup directory."""
        count = 0
        
        for item in source.rglob("*"):
            if item.is_file() and item.name != "manifest.json":
                rel_path = item.relative_to(source)
                
                # Handle compressed files
                if item.suffix == ".gz":
                    dest_file = target / rel_path.with_suffix("")
                    dest_file = Path(str(dest_file).replace(".gz", ""))
                else:
                    dest_file = target / rel_path
                
                if dest_file.exists() and not overwrite:
                    continue
                
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                if item.suffix == ".gz":
                    with gzip.open(item, 'rb') as f_in:
                        with open(dest_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(item, dest_file)
                
                count += 1
        
        return count
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        backup_path = self.backup_root / backup_id
        
        if not backup_path.exists():
            return {"valid": False, "error": "Backup not found"}
        
        manifest_file = backup_path / "manifest.json"
        if not manifest_file.exists():
            return {"valid": False, "error": "Manifest missing"}
        
        manifest = json.loads(manifest_file.read_text())
        
        # Count actual files
        actual_files = sum(1 for f in backup_path.rglob("*") 
                          if f.is_file() and f.name != "manifest.json")
        
        return {
            "valid": True,
            "backup_id": backup_id,
            "expected_files": manifest.get("files", 0),
            "actual_files": actual_files,
            "timestamp": manifest.get("timestamp"),
        }
