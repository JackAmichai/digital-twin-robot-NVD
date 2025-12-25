"""
Conversation module for multi-turn dialogue with voice-controlled robots.

Provides:
- Redis-backed conversation context persistence
- Context-aware response generation
- Slot filling for incomplete commands
- Session lifecycle management
- Coreference resolution
"""

from .context_manager import (
    MessageRole,
    ConversationMessage,
    ConversationContext,
    ConversationContextManager,
    get_context_manager,
)

from .response_generator import (
    ResponseType,
    ContextAwareResponse,
    ContextAwareResponseGenerator,
    SlotFillingManager,
)

from .session_manager import (
    SessionState,
    ProcessingResult,
    ConversationSession,
    SessionManager,
)


__all__ = [
    # Context management
    'MessageRole',
    'ConversationMessage',
    'ConversationContext',
    'ConversationContextManager',
    'get_context_manager',
    # Response generation
    'ResponseType',
    'ContextAwareResponse',
    'ContextAwareResponseGenerator',
    'SlotFillingManager',
    # Session management
    'SessionState',
    'ProcessingResult',
    'ConversationSession',
    'SessionManager',
]

__version__ = '1.0.0'
