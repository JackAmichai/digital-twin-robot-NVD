# Backup & Recovery Module

## Overview
Automated backup and disaster recovery tools.

## Files

### `backup_manager.py`
Manages backup procedures.

```python
class BackupManager:
    def add_config(config: BackupConfig)
    def run_backup(config_name) -> BackupResult
    def list_backups(config_name) -> List[BackupResult]
    def cleanup_old_backups(config_name) -> int
```

**Backup Types:**
- FULL: Complete backup
- INCREMENTAL: Changes since last backup
- DIFFERENTIAL: Changes since last full

**Storage Backends:**
- LOCAL: Filesystem
- S3: AWS S3
- GCS: Google Cloud Storage
- AZURE: Azure Blob

### `recovery.py`
Handles disaster recovery.

```python
class RecoveryManager:
    def list_available_backups() -> List[Dict]
    def restore(backup_id, target_path) -> RecoveryResult
    def verify_backup(backup_id) -> Dict
```

## Usage

```python
from backup import BackupManager, BackupConfig

config = BackupConfig(
    name="daily-backup",
    source_paths=["/data", "/config"],
    destination="/backups",
    retention_days=30,
    compression=True,
)

manager = BackupManager([config])
result = manager.run_backup("daily-backup")
print(f"Backed up {result.files_backed_up} files")
```
