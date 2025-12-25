# Feature Flags Module

## Overview
Runtime feature toggle system for controlled rollouts.

## Files

### `feature_manager.py`
Manages feature flags with targeting strategies.

```python
class FeatureManager:
    def create_flag(flag: FeatureFlag) -> FeatureFlag
    def update_flag(name, **kwargs) -> FeatureFlag
    def delete_flag(name)
    def is_enabled(flag_name, user_id, groups) -> bool
    def enable(flag_name)
    def disable(flag_name)
    def set_percentage(flag_name, percentage)
    def add_user(flag_name, user_id)
```

**Flag Types:**
- `BOOLEAN`: Simple on/off toggle
- `PERCENTAGE`: Gradual rollout (0-100%)
- `USER_LIST`: Specific users only
- `GROUP`: Target user groups

## Usage

```python
from features import FeatureManager, FeatureFlag, FlagType

manager = FeatureManager()

# Boolean flag
flag = FeatureFlag(
    name="new_ui",
    description="Enable new UI design",
    flag_type=FlagType.BOOLEAN,
    enabled=False,
)
manager.create_flag(flag)

# Percentage rollout
manager.create_flag(FeatureFlag(
    name="ml_model_v2",
    description="New ML model",
    flag_type=FlagType.PERCENTAGE,
    percentage=10,  # 10% of users
))

# Check if enabled
if manager.is_enabled("new_ui"):
    render_new_ui()
else:
    render_old_ui()

# Percentage check (deterministic per user)
if manager.is_enabled("ml_model_v2", user_id="user123"):
    use_new_model()
```

## Targeting
- Percentage rollout is deterministic (same user always gets same result)
- Uses MD5 hash of flag_name:user_id for bucketing
- Redis-backed for persistence across instances
