"""
Conversation Context Manager for Multi-Turn Dialogue.

Provides persistent conversation memory using Redis, enabling context-aware
responses across multiple turns of voice interaction with the robot.
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool


class MessageRole(str, Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """
    A single message in the conversation.
    
    Attributes:
        role: Who sent the message (user/assistant/system)
        content: The message text
        timestamp: When the message was sent
        metadata: Additional context (intent, entities, confidence, etc.)
    """
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )
    
    def to_llm_format(self) -> Dict[str, str]:
        """Convert to format expected by LLM APIs."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ConversationContext:
    """
    Complete conversation context with history and state.
    
    Attributes:
        session_id: Unique identifier for the conversation session
        user_id: Optional user identifier for personalization
        robot_id: Robot handling the conversation
        messages: List of conversation messages
        entities: Extracted entities across the conversation
        current_intent: Most recent detected intent
        task_context: Context about current robot task
        created_at: Session creation timestamp
        last_activity: Last interaction timestamp
        turn_count: Number of conversation turns
    """
    session_id: str
    robot_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    user_id: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    current_intent: Optional[str] = None
    task_context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "entities": self.entities,
            "current_intent": self.current_intent,
            "task_context": self.task_context,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "turn_count": self.turn_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            robot_id=data["robot_id"],
            user_id=data.get("user_id"),
            messages=[ConversationMessage.from_dict(m) for m in data.get("messages", [])],
            entities=data.get("entities", {}),
            current_intent=data.get("current_intent"),
            task_context=data.get("task_context", {}),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            turn_count=data.get("turn_count", 0)
        )
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """Add a new message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = time.time()
        
        if role == MessageRole.USER:
            self.turn_count += 1
        
        return message
    
    def get_recent_messages(self, count: int = 10) -> List[ConversationMessage]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []
    
    def get_llm_messages(
        self,
        max_messages: int = 20,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM API.
        
        Args:
            max_messages: Maximum number of messages to include
            include_system: Whether to include system messages
            
        Returns:
            List of messages in LLM format
        """
        messages = []
        
        for msg in self.messages[-max_messages:]:
            if not include_system and msg.role == MessageRole.SYSTEM:
                continue
            messages.append(msg.to_llm_format())
        
        return messages
    
    def update_entity(self, entity_type: str, value: Any) -> None:
        """Update or add an entity."""
        self.entities[entity_type] = {
            "value": value,
            "updated_at": time.time()
        }
    
    def get_entity(self, entity_type: str) -> Optional[Any]:
        """Get an entity value."""
        entity = self.entities.get(entity_type)
        return entity["value"] if entity else None
    
    def get_context_summary(self) -> str:
        """Generate a summary of the conversation context for LLM."""
        summary_parts = []
        
        if self.current_intent:
            summary_parts.append(f"Current intent: {self.current_intent}")
        
        if self.entities:
            entity_strs = [f"{k}: {v['value']}" for k, v in self.entities.items()]
            summary_parts.append(f"Known entities: {', '.join(entity_strs)}")
        
        if self.task_context:
            task_str = json.dumps(self.task_context, indent=2)
            summary_parts.append(f"Current task: {task_str}")
        
        return "\n".join(summary_parts) if summary_parts else "No prior context."


class ConversationContextManager:
    """
    Manager for conversation contexts with Redis persistence.
    
    Features:
    - Async Redis operations for high performance
    - Automatic session expiration
    - Context summarization for long conversations
    - Entity tracking across turns
    - Multi-robot session management
    
    Example:
        >>> manager = ConversationContextManager()
        >>> await manager.connect()
        >>> 
        >>> # Get or create session
        >>> context = await manager.get_or_create_session("robot_1", "user_123")
        >>> 
        >>> # Add user message
        >>> context.add_message(MessageRole.USER, "Move to the warehouse")
        >>> context.current_intent = "navigate"
        >>> context.update_entity("destination", "warehouse")
        >>> 
        >>> # Save context
        >>> await manager.save_context(context)
        >>> 
        >>> # Later, retrieve context
        >>> context = await manager.get_context(context.session_id)
    """
    
    # Redis key prefixes
    CONTEXT_PREFIX = "conversation:context:"
    SESSION_INDEX_PREFIX = "conversation:sessions:"
    USER_SESSIONS_PREFIX = "conversation:user:"
    ROBOT_SESSIONS_PREFIX = "conversation:robot:"
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        session_ttl_hours: int = 24,
        max_messages_per_session: int = 100,
        context_window_size: int = 20
    ):
        """
        Initialize the context manager.
        
        Args:
            redis_url: Redis connection URL
            session_ttl_hours: Hours before session expires
            max_messages_per_session: Maximum messages to store per session
            context_window_size: Number of messages to include in LLM context
        """
        self.redis_url = redis_url
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.max_messages = max_messages_per_session
        self.context_window = context_window_size
        
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        self._pool = ConnectionPool.from_url(
            self.redis_url,
            max_connections=20,
            decode_responses=True
        )
        self._redis = redis.Redis(connection_pool=self._pool)
        
        # Verify connection
        await self._redis.ping()
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
    
    def _generate_session_id(self, robot_id: str, user_id: Optional[str] = None) -> str:
        """Generate a unique session ID."""
        timestamp = str(time.time())
        components = [robot_id, user_id or "anonymous", timestamp]
        hash_input = ":".join(components)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _context_key(self, session_id: str) -> str:
        """Get Redis key for a context."""
        return f"{self.CONTEXT_PREFIX}{session_id}"
    
    async def create_session(
        self,
        robot_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ConversationContext:
        """
        Create a new conversation session.
        
        Args:
            robot_id: Robot handling the conversation
            user_id: Optional user identifier
            system_prompt: Optional system message to initialize context
            
        Returns:
            New ConversationContext
        """
        session_id = self._generate_session_id(robot_id, user_id)
        
        context = ConversationContext(
            session_id=session_id,
            robot_id=robot_id,
            user_id=user_id
        )
        
        # Add system prompt if provided
        if system_prompt:
            context.add_message(MessageRole.SYSTEM, system_prompt)
        
        # Save to Redis
        await self.save_context(context)
        
        # Index the session
        await self._index_session(context)
        
        return context
    
    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Retrieve a conversation context by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext or None if not found
        """
        key = self._context_key(session_id)
        data = await self._redis.get(key)
        
        if not data:
            return None
        
        context_dict = json.loads(data)
        return ConversationContext.from_dict(context_dict)
    
    async def save_context(self, context: ConversationContext) -> None:
        """
        Save a conversation context to Redis.
        
        Args:
            context: ConversationContext to save
        """
        # Trim messages if exceeding limit
        if len(context.messages) > self.max_messages:
            # Keep system messages and recent messages
            system_msgs = [m for m in context.messages if m.role == MessageRole.SYSTEM]
            other_msgs = [m for m in context.messages if m.role != MessageRole.SYSTEM]
            context.messages = system_msgs + other_msgs[-(self.max_messages - len(system_msgs)):]
        
        key = self._context_key(context.session_id)
        data = json.dumps(context.to_dict())
        
        # Save with TTL
        ttl_seconds = int(self.session_ttl.total_seconds())
        await self._redis.setex(key, ttl_seconds, data)
    
    async def _index_session(self, context: ConversationContext) -> None:
        """Index session for lookup by user/robot."""
        ttl_seconds = int(self.session_ttl.total_seconds())
        
        # Index by robot
        robot_key = f"{self.ROBOT_SESSIONS_PREFIX}{context.robot_id}"
        await self._redis.zadd(robot_key, {context.session_id: context.created_at})
        await self._redis.expire(robot_key, ttl_seconds)
        
        # Index by user if provided
        if context.user_id:
            user_key = f"{self.USER_SESSIONS_PREFIX}{context.user_id}"
            await self._redis.zadd(user_key, {context.session_id: context.created_at})
            await self._redis.expire(user_key, ttl_seconds)
    
    async def get_or_create_session(
        self,
        robot_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reuse_recent_minutes: int = 30
    ) -> ConversationContext:
        """
        Get recent session or create new one.
        
        If a session exists for the robot/user within the reuse window,
        returns that session. Otherwise creates a new one.
        
        Args:
            robot_id: Robot identifier
            user_id: Optional user identifier
            system_prompt: System prompt for new sessions
            reuse_recent_minutes: Minutes to look back for existing session
            
        Returns:
            ConversationContext (existing or new)
        """
        # Look for recent session
        recent_session = await self.get_recent_session(
            robot_id, user_id, minutes=reuse_recent_minutes
        )
        
        if recent_session:
            return recent_session
        
        # Create new session
        return await self.create_session(robot_id, user_id, system_prompt)
    
    async def get_recent_session(
        self,
        robot_id: str,
        user_id: Optional[str] = None,
        minutes: int = 30
    ) -> Optional[ConversationContext]:
        """
        Get the most recent session for a robot/user.
        
        Args:
            robot_id: Robot identifier
            user_id: Optional user identifier
            minutes: Look back window in minutes
            
        Returns:
            Most recent ConversationContext or None
        """
        cutoff_time = time.time() - (minutes * 60)
        
        # Get sessions for this robot
        robot_key = f"{self.ROBOT_SESSIONS_PREFIX}{robot_id}"
        sessions = await self._redis.zrangebyscore(
            robot_key, cutoff_time, "+inf", withscores=True
        )
        
        if not sessions:
            return None
        
        # Get most recent session
        for session_id, _ in reversed(sessions):
            context = await self.get_context(session_id)
            if context:
                # If user_id specified, check it matches
                if user_id and context.user_id != user_id:
                    continue
                return context
        
        return None
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[ConversationContext]:
        """
        Get recent sessions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum sessions to return
            
        Returns:
            List of ConversationContext
        """
        user_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        session_ids = await self._redis.zrevrange(user_key, 0, limit - 1)
        
        contexts = []
        for session_id in session_ids:
            context = await self.get_context(session_id)
            if context:
                contexts.append(context)
        
        return contexts
    
    async def add_user_message(
        self,
        session_id: str,
        content: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None
    ) -> Optional[ConversationContext]:
        """
        Add a user message to a session.
        
        Args:
            session_id: Session identifier
            content: Message content (transcript)
            intent: Detected intent
            entities: Extracted entities
            confidence: ASR confidence score
            
        Returns:
            Updated ConversationContext or None if session not found
        """
        context = await self.get_context(session_id)
        if not context:
            return None
        
        # Build metadata
        metadata = {}
        if confidence is not None:
            metadata["asr_confidence"] = confidence
        if intent:
            metadata["intent"] = intent
            context.current_intent = intent
        if entities:
            metadata["entities"] = entities
            for entity_type, value in entities.items():
                context.update_entity(entity_type, value)
        
        # Add message
        context.add_message(MessageRole.USER, content, metadata)
        
        # Save
        await self.save_context(context)
        
        return context
    
    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        action_taken: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ConversationContext]:
        """
        Add an assistant response to a session.
        
        Args:
            session_id: Session identifier
            content: Response content
            action_taken: Action executed by the robot
            metadata: Additional metadata
            
        Returns:
            Updated ConversationContext or None if session not found
        """
        context = await self.get_context(session_id)
        if not context:
            return None
        
        # Build metadata
        msg_metadata = metadata or {}
        if action_taken:
            msg_metadata["action"] = action_taken
        
        # Add message
        context.add_message(MessageRole.ASSISTANT, content, msg_metadata)
        
        # Save
        await self.save_context(context)
        
        return context
    
    async def update_task_context(
        self,
        session_id: str,
        task_context: Dict[str, Any]
    ) -> Optional[ConversationContext]:
        """
        Update the task context for a session.
        
        Args:
            session_id: Session identifier
            task_context: Task context to merge
            
        Returns:
            Updated ConversationContext or None
        """
        context = await self.get_context(session_id)
        if not context:
            return None
        
        context.task_context.update(task_context)
        await self.save_context(context)
        
        return context
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        key = self._context_key(session_id)
        deleted = await self._redis.delete(key)
        return deleted > 0
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired session indices.
        
        Returns:
            Number of sessions cleaned up
        """
        # Redis TTL handles context expiration automatically
        # This cleans up the index sorted sets
        cleaned = 0
        cutoff_time = time.time() - self.session_ttl.total_seconds()
        
        # Get all robot session keys
        robot_keys = await self._redis.keys(f"{self.ROBOT_SESSIONS_PREFIX}*")
        for key in robot_keys:
            removed = await self._redis.zremrangebyscore(key, "-inf", cutoff_time)
            cleaned += removed
        
        # Get all user session keys
        user_keys = await self._redis.keys(f"{self.USER_SESSIONS_PREFIX}*")
        for key in user_keys:
            removed = await self._redis.zremrangebyscore(key, "-inf", cutoff_time)
            cleaned += removed
        
        return cleaned


# Singleton instance
_context_manager: Optional[ConversationContextManager] = None


async def get_context_manager(
    redis_url: str = "redis://localhost:6379"
) -> ConversationContextManager:
    """
    Get or create the global context manager.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        ConversationContextManager instance
    """
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ConversationContextManager(redis_url=redis_url)
        await _context_manager.connect()
    
    return _context_manager


__all__ = [
    'MessageRole',
    'ConversationMessage',
    'ConversationContext',
    'ConversationContextManager',
    'get_context_manager',
]
