"""
Feature Manager - Runtime feature toggle system.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum


class FlagType(Enum):
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    GROUP = "group"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    description: str
    flag_type: FlagType = FlagType.BOOLEAN
    enabled: bool = False
    percentage: int = 0  # For percentage rollout
    allowed_users: Set[str] = field(default_factory=set)
    allowed_groups: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "flag_type": self.flag_type.value,
            "enabled": self.enabled,
            "percentage": self.percentage,
            "allowed_users": list(self.allowed_users),
            "allowed_groups": list(self.allowed_groups),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FeatureManager:
    """
    Manages feature flags with various targeting strategies.
    """
    
    def __init__(self, redis_client=None):
        self.flags: Dict[str, FeatureFlag] = {}
        self.redis = redis_client
        self._load_flags()
    
    def _load_flags(self) -> None:
        """Load flags from Redis or config."""
        if self.redis:
            try:
                data = self.redis.get("feature_flags")
                if data:
                    flags_data = json.loads(data)
                    for name, flag_dict in flags_data.items():
                        self.flags[name] = self._dict_to_flag(flag_dict)
            except Exception:
                pass
    
    def _save_flags(self) -> None:
        """Save flags to Redis."""
        if self.redis:
            data = {name: flag.to_dict() for name, flag in self.flags.items()}
            self.redis.set("feature_flags", json.dumps(data))
    
    def _dict_to_flag(self, d: Dict) -> FeatureFlag:
        """Convert dict to FeatureFlag."""
        return FeatureFlag(
            name=d["name"],
            description=d["description"],
            flag_type=FlagType(d["flag_type"]),
            enabled=d["enabled"],
            percentage=d.get("percentage", 0),
            allowed_users=set(d.get("allowed_users", [])),
            allowed_groups=set(d.get("allowed_groups", [])),
        )
    
    def create_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Create a new feature flag."""
        self.flags[flag.name] = flag
        self._save_flags()
        return flag
    
    def update_flag(self, name: str, **kwargs) -> FeatureFlag:
        """Update a feature flag."""
        if name not in self.flags:
            raise ValueError(f"Flag not found: {name}")
        
        flag = self.flags[name]
        for key, value in kwargs.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        flag.updated_at = datetime.now()
        
        self._save_flags()
        return flag
    
    def delete_flag(self, name: str) -> None:
        """Delete a feature flag."""
        if name in self.flags:
            del self.flags[name]
            self._save_flags()
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        groups: Optional[List[str]] = None
    ) -> bool:
        """
        Check if a feature is enabled for a user.
        
        Args:
            flag_name: Name of the feature flag
            user_id: Optional user ID for targeting
            groups: Optional list of user groups
        """
        if flag_name not in self.flags:
            return False
        
        flag = self.flags[flag_name]
        
        # Boolean flag
        if flag.flag_type == FlagType.BOOLEAN:
            return flag.enabled
        
        # User list targeting
        if flag.flag_type == FlagType.USER_LIST:
            return user_id in flag.allowed_users if user_id else False
        
        # Group targeting
        if flag.flag_type == FlagType.GROUP:
            if not groups:
                return False
            return bool(flag.allowed_groups & set(groups))
        
        # Percentage rollout
        if flag.flag_type == FlagType.PERCENTAGE:
            if not user_id:
                return False
            return self._check_percentage(flag_name, user_id, flag.percentage)
        
        return False
    
    def _check_percentage(self, flag_name: str, user_id: str, percentage: int) -> bool:
        """Deterministic percentage check based on user ID."""
        hash_input = f"{flag_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return (hash_value % 100) < percentage
    
    def enable(self, flag_name: str) -> None:
        """Enable a boolean flag."""
        self.update_flag(flag_name, enabled=True)
    
    def disable(self, flag_name: str) -> None:
        """Disable a boolean flag."""
        self.update_flag(flag_name, enabled=False)
    
    def set_percentage(self, flag_name: str, percentage: int) -> None:
        """Set percentage for rollout."""
        self.update_flag(flag_name, percentage=min(100, max(0, percentage)))
    
    def add_user(self, flag_name: str, user_id: str) -> None:
        """Add user to allowed list."""
        if flag_name in self.flags:
            self.flags[flag_name].allowed_users.add(user_id)
            self._save_flags()
    
    def remove_user(self, flag_name: str, user_id: str) -> None:
        """Remove user from allowed list."""
        if flag_name in self.flags:
            self.flags[flag_name].allowed_users.discard(user_id)
            self._save_flags()
    
    def get_all_flags(self) -> Dict[str, Dict]:
        """Get all feature flags."""
        return {name: flag.to_dict() for name, flag in self.flags.items()}
    
    def get_enabled_flags(self, user_id: str = None, groups: List[str] = None) -> List[str]:
        """Get list of enabled flags for a user."""
        return [
            name for name in self.flags
            if self.is_enabled(name, user_id, groups)
        ]
