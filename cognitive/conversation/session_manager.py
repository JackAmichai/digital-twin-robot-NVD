"""
Conversation Session Manager for Voice-Controlled Robots.

Provides high-level session management integrating ASR, NLU, context,
and response generation for seamless multi-turn conversations.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import logging

from .context_manager import (
    ConversationContext,
    ConversationContextManager,
    MessageRole,
    get_context_manager
)
from .response_generator import (
    ContextAwareResponseGenerator,
    ContextAwareResponse,
    ResponseType,
    SlotFillingManager
)


logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """State of a conversation session."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_SLOT = "awaiting_slot"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """
    Result of processing a voice input.
    
    Attributes:
        response: Generated response for TTS
        action: Action to execute (if any)
        action_params: Parameters for the action
        session_state: New session state
        requires_input: Whether more user input is needed
        error: Error message if processing failed
    """
    response: ContextAwareResponse
    action: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    session_state: SessionState = SessionState.IDLE
    requires_input: bool = False
    error: Optional[str] = None


class ConversationSession:
    """
    A single conversation session with a robot.
    
    Manages the complete lifecycle of a conversation including:
    - Session state management
    - Context tracking
    - Slot filling for incomplete commands
    - Confirmation handling
    - Action execution coordination
    
    Example:
        >>> session = await ConversationSession.create("robot_1", "user_123")
        >>> 
        >>> # Process voice input
        >>> result = await session.process_input(
        ...     transcript="Take the box to the warehouse",
        ...     intent="navigate",
        ...     entities={"object": "box", "destination": "warehouse"}
        ... )
        >>> 
        >>> # Get response for TTS
        >>> print(result.response.text)
        >>> # "I'll take the box to the warehouse."
        >>> 
        >>> # Execute action if provided
        >>> if result.action:
        ...     await robot.execute(result.action, result.action_params)
    """
    
    def __init__(
        self,
        context: ConversationContext,
        context_manager: ConversationContextManager,
        response_generator: ContextAwareResponseGenerator,
        slot_manager: SlotFillingManager
    ):
        """
        Initialize conversation session.
        
        Use ConversationSession.create() for async initialization.
        """
        self.context = context
        self._context_manager = context_manager
        self._response_generator = response_generator
        self._slot_manager = slot_manager
        
        self._state = SessionState.IDLE
        self._pending_intent: Optional[str] = None
        self._pending_entities: Dict[str, Any] = {}
        self._pending_slot: Optional[str] = None
        self._confirmation_callback: Optional[Callable] = None
        
        self._last_activity = time.time()
        self._timeout_seconds = 300  # 5 minute timeout
    
    @classmethod
    async def create(
        cls,
        robot_id: str,
        user_id: Optional[str] = None,
        llm_client: Any = None,
        context_manager: Optional[ConversationContextManager] = None,
        system_prompt: Optional[str] = None
    ) -> "ConversationSession":
        """
        Create a new conversation session.
        
        Args:
            robot_id: Robot handling the conversation
            user_id: Optional user identifier
            llm_client: LLM client for response generation
            context_manager: Optional context manager
            system_prompt: Optional system prompt
            
        Returns:
            Initialized ConversationSession
        """
        # Get or create context manager
        manager = context_manager or await get_context_manager()
        
        # Get or create conversation context
        context = await manager.get_or_create_session(
            robot_id=robot_id,
            user_id=user_id,
            system_prompt=system_prompt
        )
        
        # Create response generator
        response_gen = ContextAwareResponseGenerator(
            llm_client=llm_client,
            context_manager=manager
        )
        
        # Create slot manager
        slot_manager = SlotFillingManager()
        
        return cls(context, manager, response_gen, slot_manager)
    
    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self.context.session_id
    
    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active (not timed out)."""
        return (time.time() - self._last_activity) < self._timeout_seconds
    
    async def process_input(
        self,
        transcript: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None
    ) -> ProcessingResult:
        """
        Process a voice input and generate response.
        
        Args:
            transcript: ASR transcript
            intent: Detected intent
            entities: Extracted entities
            confidence: ASR confidence score
            
        Returns:
            ProcessingResult with response and action
        """
        self._last_activity = time.time()
        self._state = SessionState.PROCESSING
        
        try:
            # Add user message to context
            await self._context_manager.add_user_message(
                self.context.session_id,
                transcript,
                intent=intent,
                entities=entities,
                confidence=confidence
            )
            
            # Refresh context
            self.context = await self._context_manager.get_context(
                self.context.session_id
            )
            
            # Handle based on current state
            if self._pending_slot:
                return await self._handle_slot_filling(transcript, entities)
            
            if self._state == SessionState.AWAITING_CONFIRMATION:
                return await self._handle_confirmation(transcript)
            
            # Normal processing
            return await self._process_new_input(
                transcript, intent, entities
            )
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            self._state = SessionState.ERROR
            return ProcessingResult(
                response=ContextAwareResponse(
                    text="I'm sorry, I encountered an error. Please try again.",
                    response_type=ResponseType.ERROR
                ),
                session_state=SessionState.ERROR,
                error=str(e)
            )
    
    async def _process_new_input(
        self,
        transcript: str,
        intent: Optional[str],
        entities: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process a new input (not continuation)."""
        entities = entities or {}
        
        # Check for required slots
        if intent:
            all_filled, missing_prompt = self._slot_manager.check_slots(
                intent, entities
            )
            
            if not all_filled:
                # Need to fill slots
                self._pending_intent = intent
                self._pending_entities = entities
                self._state = SessionState.AWAITING_SLOT
                
                return ProcessingResult(
                    response=ContextAwareResponse(
                        text=missing_prompt,
                        response_type=ResponseType.CLARIFICATION,
                        follow_up_prompt=missing_prompt
                    ),
                    session_state=SessionState.AWAITING_SLOT,
                    requires_input=True
                )
        
        # Generate response
        response = await self._response_generator.generate_response(
            self.context,
            transcript,
            intent=intent,
            entities=entities
        )
        
        # Determine new state and action
        new_state = SessionState.IDLE
        action = None
        action_params = None
        requires_input = False
        
        if response.requires_confirmation:
            new_state = SessionState.AWAITING_CONFIRMATION
            self._pending_intent = response.action_to_execute
            self._pending_entities = response.action_params or {}
            requires_input = True
        elif response.action_to_execute:
            new_state = SessionState.EXECUTING
            action = response.action_to_execute
            action_params = response.action_params
        elif response.response_type == ResponseType.CLARIFICATION:
            requires_input = True
        
        self._state = new_state
        
        return ProcessingResult(
            response=response,
            action=action,
            action_params=action_params,
            session_state=new_state,
            requires_input=requires_input
        )
    
    async def _handle_slot_filling(
        self,
        transcript: str,
        entities: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Handle input when waiting for slot value."""
        entities = entities or {}
        
        # Merge with pending entities
        self._pending_entities.update(entities)
        
        # If no entities extracted, use transcript as slot value
        if not entities and self._pending_slot:
            self._pending_entities[self._pending_slot] = transcript
        
        # Check if all slots now filled
        all_filled, missing_prompt = self._slot_manager.check_slots(
            self._pending_intent,
            self._pending_entities
        )
        
        if not all_filled:
            # Still need more slots
            return ProcessingResult(
                response=ContextAwareResponse(
                    text=missing_prompt,
                    response_type=ResponseType.CLARIFICATION,
                    follow_up_prompt=missing_prompt
                ),
                session_state=SessionState.AWAITING_SLOT,
                requires_input=True
            )
        
        # All slots filled, process command
        normalized_entities = self._slot_manager.normalize_entities(
            self._pending_intent,
            self._pending_entities
        )
        
        # Generate response for complete command
        response = await self._response_generator.generate_response(
            self.context,
            transcript,
            intent=self._pending_intent,
            entities=normalized_entities
        )
        
        # Clear pending state
        intent = self._pending_intent
        params = normalized_entities
        self._pending_intent = None
        self._pending_entities = {}
        self._pending_slot = None
        self._state = SessionState.EXECUTING
        
        return ProcessingResult(
            response=response,
            action=intent,
            action_params=params,
            session_state=SessionState.EXECUTING
        )
    
    async def _handle_confirmation(
        self,
        transcript: str
    ) -> ProcessingResult:
        """Handle confirmation response."""
        transcript_lower = transcript.lower()
        
        # Check for affirmative
        affirmative = any(word in transcript_lower for word in [
            "yes", "yeah", "yep", "sure", "okay", "ok", "confirm",
            "do it", "go ahead", "proceed", "correct"
        ])
        
        # Check for negative
        negative = any(word in transcript_lower for word in [
            "no", "nope", "cancel", "stop", "don't", "never mind",
            "abort", "wait"
        ])
        
        if affirmative:
            # Execute pending action
            action = self._pending_intent
            params = self._pending_entities
            
            response = ContextAwareResponse(
                text="Executing now.",
                response_type=ResponseType.ACKNOWLEDGMENT,
                action_to_execute=action,
                action_params=params
            )
            
            # Clear pending state
            self._pending_intent = None
            self._pending_entities = {}
            self._state = SessionState.EXECUTING
            
            return ProcessingResult(
                response=response,
                action=action,
                action_params=params,
                session_state=SessionState.EXECUTING
            )
        
        elif negative:
            # Cancel pending action
            response = ContextAwareResponse(
                text="Okay, I've cancelled that action. What would you like me to do instead?",
                response_type=ResponseType.ACKNOWLEDGMENT
            )
            
            self._pending_intent = None
            self._pending_entities = {}
            self._state = SessionState.IDLE
            
            return ProcessingResult(
                response=response,
                session_state=SessionState.IDLE,
                requires_input=True
            )
        
        else:
            # Unclear response, ask again
            action_desc = f"{self._pending_intent} with {self._pending_entities}"
            response = ContextAwareResponse(
                text=f"I'm sorry, I didn't understand. Should I {action_desc}? Please say yes or no.",
                response_type=ResponseType.CONFIRMATION
            )
            
            return ProcessingResult(
                response=response,
                session_state=SessionState.AWAITING_CONFIRMATION,
                requires_input=True
            )
    
    async def report_action_result(
        self,
        success: bool,
        details: Optional[str] = None
    ) -> ContextAwareResponse:
        """
        Report the result of an executed action.
        
        Args:
            success: Whether the action succeeded
            details: Additional details about the result
            
        Returns:
            Response for TTS
        """
        self._last_activity = time.time()
        
        if success:
            response_text = details or "Task completed successfully."
            response_type = ResponseType.ACTION_RESULT
        else:
            response_text = details or "I wasn't able to complete that task."
            response_type = ResponseType.ERROR
        
        # Add to context
        await self._context_manager.add_assistant_message(
            self.context.session_id,
            response_text,
            action_taken="report_result",
            metadata={"success": success}
        )
        
        self._state = SessionState.IDLE
        
        return ContextAwareResponse(
            text=response_text,
            response_type=response_type
        )
    
    async def end_session(self) -> None:
        """End the conversation session."""
        self._state = SessionState.IDLE
        # Context remains in Redis for history, but session is logically ended
        logger.info(f"Session {self.session_id} ended")


class SessionManager:
    """
    Manages multiple conversation sessions across robots and users.
    
    Example:
        >>> manager = SessionManager(llm_client)
        >>> await manager.initialize()
        >>> 
        >>> # Get or create session for a robot
        >>> session = await manager.get_session("robot_1", "user_123")
        >>> 
        >>> # Process input
        >>> result = await session.process_input("Go to the warehouse")
    """
    
    def __init__(
        self,
        llm_client: Any,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize session manager.
        
        Args:
            llm_client: LLM client for response generation
            redis_url: Redis connection URL
        """
        self.llm_client = llm_client
        self.redis_url = redis_url
        
        self._context_manager: Optional[ConversationContextManager] = None
        self._sessions: Dict[str, ConversationSession] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the session manager."""
        if self._initialized:
            return
        
        self._context_manager = ConversationContextManager(
            redis_url=self.redis_url
        )
        await self._context_manager.connect()
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        if self._context_manager:
            await self._context_manager.disconnect()
        self._sessions.clear()
        self._initialized = False
    
    async def get_session(
        self,
        robot_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ConversationSession:
        """
        Get or create a conversation session.
        
        Args:
            robot_id: Robot identifier
            user_id: Optional user identifier
            system_prompt: Optional system prompt for new sessions
            
        Returns:
            ConversationSession instance
        """
        if not self._initialized:
            await self.initialize()
        
        # Create session key
        session_key = f"{robot_id}:{user_id or 'anonymous'}"
        
        # Check for existing active session
        if session_key in self._sessions:
            session = self._sessions[session_key]
            if session.is_active:
                return session
        
        # Create new session
        session = await ConversationSession.create(
            robot_id=robot_id,
            user_id=user_id,
            llm_client=self.llm_client,
            context_manager=self._context_manager,
            system_prompt=system_prompt
        )
        
        self._sessions[session_key] = session
        return session
    
    async def end_session(
        self,
        robot_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """End a specific session."""
        session_key = f"{robot_id}:{user_id or 'anonymous'}"
        
        if session_key in self._sessions:
            await self._sessions[session_key].end_session()
            del self._sessions[session_key]
    
    async def cleanup_inactive_sessions(self) -> int:
        """
        Clean up inactive sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        inactive = [
            key for key, session in self._sessions.items()
            if not session.is_active
        ]
        
        for key in inactive:
            await self._sessions[key].end_session()
            del self._sessions[key]
        
        return len(inactive)


__all__ = [
    'SessionState',
    'ProcessingResult',
    'ConversationSession',
    'SessionManager',
]
