"""
Context-Aware Response Generator.

Generates context-aware responses using conversation history and LLM,
enabling natural multi-turn dialogue with the robot.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from .context_manager import (
    ConversationContext,
    ConversationContextManager,
    MessageRole,
    get_context_manager
)


class ResponseType(str, Enum):
    """Type of response generated."""
    ACKNOWLEDGMENT = "acknowledgment"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    INFORMATION = "information"
    ACTION_RESULT = "action_result"
    ERROR = "error"


@dataclass
class ContextAwareResponse:
    """
    A context-aware response with metadata.
    
    Attributes:
        text: Response text for TTS
        response_type: Category of response
        action_to_execute: Optional action to execute
        requires_confirmation: Whether to wait for user confirmation
        follow_up_prompt: Optional prompt for follow-up
        entities_updated: Entities that were updated
        confidence: Confidence in the response
    """
    text: str
    response_type: ResponseType
    action_to_execute: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    requires_confirmation: bool = False
    follow_up_prompt: Optional[str] = None
    entities_updated: Dict[str, Any] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.entities_updated is None:
            self.entities_updated = {}


class ContextAwareResponseGenerator:
    """
    Generates context-aware responses using conversation history.
    
    Features:
    - Multi-turn context understanding
    - Coreference resolution (it, that, there, etc.)
    - Intent continuation
    - Entity slot filling
    - Clarification handling
    
    Example:
        >>> generator = ContextAwareResponseGenerator(llm_client)
        >>> 
        >>> # User: "Move to the warehouse"
        >>> response = await generator.generate_response(
        ...     context, "Move to the warehouse", intent="navigate"
        ... )
        >>> # Response: "I'll navigate to the warehouse now."
        >>> 
        >>> # User: "Actually, go to the loading dock instead"
        >>> response = await generator.generate_response(
        ...     context, "Actually, go to the loading dock instead"
        ... )
        >>> # Response: "Understood. Changing destination to the loading dock."
    """
    
    # System prompt for context-aware responses
    SYSTEM_PROMPT = """You are a helpful robot assistant in an industrial facility.
You control a mobile robot that can navigate, pick up objects, and perform tasks.

Guidelines:
- Be concise and clear in responses (1-2 sentences)
- Acknowledge commands before executing
- Ask for clarification if the command is ambiguous
- Reference previous context when relevant
- Use natural, conversational language
- Report task completion or errors clearly

Available actions:
- navigate: Move to a location (requires: destination)
- pick: Pick up an object (requires: object_name, optional: location)
- place: Place held object (requires: destination)
- inspect: Inspect an area or object (requires: target)
- stop: Stop current action
- status: Report current status

When responding:
1. If an action is requested, confirm it
2. If information is missing, ask for it
3. If referring to previous context, acknowledge it
4. Keep responses under 30 words when possible"""

    def __init__(
        self,
        llm_client: Any,
        context_manager: Optional[ConversationContextManager] = None,
        max_context_messages: int = 10
    ):
        """
        Initialize the response generator.
        
        Args:
            llm_client: LLM client for generating responses
            context_manager: Optional context manager (will use singleton if not provided)
            max_context_messages: Maximum messages to include in LLM context
        """
        self.llm_client = llm_client
        self._context_manager = context_manager
        self.max_context_messages = max_context_messages
        
        # Coreference patterns
        self._coreference_patterns = {
            "it": self._resolve_it,
            "that": self._resolve_that,
            "there": self._resolve_there,
            "this": self._resolve_this,
            "them": self._resolve_them,
        }
        
        # Intent continuation patterns
        self._continuation_patterns = [
            "instead", "actually", "no", "change", "different",
            "also", "and", "then", "after that", "next"
        ]
    
    async def get_context_manager(self) -> ConversationContextManager:
        """Get the context manager, using singleton if not set."""
        if self._context_manager is None:
            self._context_manager = await get_context_manager()
        return self._context_manager
    
    async def generate_response(
        self,
        context: ConversationContext,
        user_input: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        action_result: Optional[Dict[str, Any]] = None
    ) -> ContextAwareResponse:
        """
        Generate a context-aware response.
        
        Args:
            context: Current conversation context
            user_input: User's input text
            intent: Detected intent (if available)
            entities: Extracted entities (if available)
            action_result: Result of previous action (if any)
            
        Returns:
            ContextAwareResponse with text and metadata
        """
        # Resolve coreferences in user input
        resolved_input = await self._resolve_coreferences(context, user_input)
        
        # Check for intent continuation
        is_continuation = self._is_intent_continuation(user_input)
        
        # Determine effective intent
        effective_intent = intent
        if is_continuation and not intent and context.current_intent:
            effective_intent = context.current_intent
        
        # Build messages for LLM
        messages = self._build_llm_messages(
            context, resolved_input, effective_intent, entities, action_result
        )
        
        # Generate response using LLM
        llm_response = await self._call_llm(messages)
        
        # Parse response for action and metadata
        response = self._parse_llm_response(
            llm_response, effective_intent, entities
        )
        
        # Update context with assistant response
        manager = await self.get_context_manager()
        await manager.add_assistant_message(
            context.session_id,
            response.text,
            action_taken=response.action_to_execute,
            metadata={
                "response_type": response.response_type.value,
                "resolved_input": resolved_input if resolved_input != user_input else None
            }
        )
        
        return response
    
    async def _resolve_coreferences(
        self,
        context: ConversationContext,
        user_input: str
    ) -> str:
        """
        Resolve coreferences in user input using conversation context.
        
        Args:
            context: Conversation context
            user_input: User's input text
            
        Returns:
            Input with coreferences resolved
        """
        resolved = user_input.lower()
        
        for pattern, resolver in self._coreference_patterns.items():
            if pattern in resolved:
                replacement = resolver(context)
                if replacement:
                    # Simple replacement - in production, use NLP for better resolution
                    resolved = resolved.replace(pattern, replacement)
        
        return resolved
    
    def _resolve_it(self, context: ConversationContext) -> Optional[str]:
        """Resolve 'it' to most recent object entity."""
        for entity_type in ["object", "item", "target"]:
            value = context.get_entity(entity_type)
            if value:
                return str(value)
        return None
    
    def _resolve_that(self, context: ConversationContext) -> Optional[str]:
        """Resolve 'that' to most recent object or location."""
        for entity_type in ["object", "location", "destination", "target"]:
            value = context.get_entity(entity_type)
            if value:
                return str(value)
        return None
    
    def _resolve_there(self, context: ConversationContext) -> Optional[str]:
        """Resolve 'there' to most recent location."""
        for entity_type in ["location", "destination", "area"]:
            value = context.get_entity(entity_type)
            if value:
                return str(value)
        return None
    
    def _resolve_this(self, context: ConversationContext) -> Optional[str]:
        """Resolve 'this' to current task context."""
        if context.task_context:
            return context.task_context.get("current_object") or \
                   context.task_context.get("target")
        return None
    
    def _resolve_them(self, context: ConversationContext) -> Optional[str]:
        """Resolve 'them' to most recent plural entity."""
        value = context.get_entity("objects")
        if value and isinstance(value, list):
            return ", ".join(value)
        return None
    
    def _is_intent_continuation(self, user_input: str) -> bool:
        """Check if input is continuing a previous intent."""
        input_lower = user_input.lower()
        return any(pattern in input_lower for pattern in self._continuation_patterns)
    
    def _build_llm_messages(
        self,
        context: ConversationContext,
        user_input: str,
        intent: Optional[str],
        entities: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Build message list for LLM API."""
        messages = []
        
        # System prompt
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        
        # Context summary if available
        context_summary = context.get_context_summary()
        if context_summary != "No prior context.":
            messages.append({
                "role": "system",
                "content": f"Current context:\n{context_summary}"
            })
        
        # Action result if available
        if action_result:
            status = action_result.get("status", "unknown")
            details = action_result.get("details", "")
            messages.append({
                "role": "system",
                "content": f"Previous action result: {status}. {details}"
            })
        
        # Conversation history
        history = context.get_llm_messages(
            max_messages=self.max_context_messages,
            include_system=False
        )
        messages.extend(history)
        
        # Current user input with metadata
        user_content = user_input
        if intent or entities:
            metadata_parts = []
            if intent:
                metadata_parts.append(f"intent={intent}")
            if entities:
                entity_str = ", ".join(f"{k}={v}" for k, v in entities.items())
                metadata_parts.append(f"entities=[{entity_str}]")
            user_content = f"{user_input}\n[Detected: {', '.join(metadata_parts)}]"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM to generate response."""
        try:
            # Assuming LLM client has a generate method
            # Adapt this to your specific LLM client interface
            response = await self.llm_client.generate(
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
            return response.get("content", response.get("text", ""))
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"
    
    def _parse_llm_response(
        self,
        llm_response: str,
        intent: Optional[str],
        entities: Optional[Dict[str, Any]]
    ) -> ContextAwareResponse:
        """Parse LLM response into structured format."""
        response_text = llm_response.strip()
        
        # Determine response type based on content
        response_type = self._classify_response_type(response_text)
        
        # Determine if action should be executed
        action_to_execute = None
        action_params = None
        requires_confirmation = False
        
        if intent and response_type == ResponseType.ACKNOWLEDGMENT:
            action_to_execute = intent
            action_params = entities or {}
            
            # Check if confirmation is needed
            if any(phrase in response_text.lower() for phrase in [
                "should i", "would you like", "confirm", "are you sure"
            ]):
                requires_confirmation = True
        
        # Check for clarification
        follow_up_prompt = None
        if response_type == ResponseType.CLARIFICATION:
            # Extract the question from response
            if "?" in response_text:
                follow_up_prompt = response_text
        
        return ContextAwareResponse(
            text=response_text,
            response_type=response_type,
            action_to_execute=action_to_execute,
            action_params=action_params,
            requires_confirmation=requires_confirmation,
            follow_up_prompt=follow_up_prompt,
            entities_updated=entities or {}
        )
    
    def _classify_response_type(self, response: str) -> ResponseType:
        """Classify the type of response."""
        response_lower = response.lower()
        
        # Check for clarification (questions)
        if "?" in response and any(w in response_lower for w in [
            "which", "where", "what", "could you", "can you specify"
        ]):
            return ResponseType.CLARIFICATION
        
        # Check for confirmation request
        if any(phrase in response_lower for phrase in [
            "should i", "would you like me to", "confirm"
        ]):
            return ResponseType.CONFIRMATION
        
        # Check for error
        if any(phrase in response_lower for phrase in [
            "error", "cannot", "unable", "failed", "sorry"
        ]):
            return ResponseType.ERROR
        
        # Check for information
        if any(phrase in response_lower for phrase in [
            "currently", "status", "battery", "located", "position"
        ]):
            return ResponseType.INFORMATION
        
        # Check for action result
        if any(phrase in response_lower for phrase in [
            "completed", "done", "finished", "arrived", "picked up"
        ]):
            return ResponseType.ACTION_RESULT
        
        # Default to acknowledgment
        return ResponseType.ACKNOWLEDGMENT


class SlotFillingManager:
    """
    Manages slot filling for multi-turn commands.
    
    Tracks required parameters and prompts user for missing information.
    
    Example:
        >>> manager = SlotFillingManager()
        >>> 
        >>> # Define required slots for navigate intent
        >>> slots = manager.get_required_slots("navigate")
        >>> # Returns: {"destination": None}
        >>> 
        >>> # Check if all slots filled
        >>> filled = manager.check_slots("navigate", {"destination": "warehouse"})
        >>> # Returns: True
    """
    
    # Required slots per intent
    INTENT_SLOTS = {
        "navigate": {
            "destination": {
                "required": True,
                "prompt": "Where would you like me to go?",
                "aliases": ["location", "place", "area"]
            }
        },
        "pick": {
            "object_name": {
                "required": True,
                "prompt": "What would you like me to pick up?",
                "aliases": ["object", "item", "thing"]
            },
            "location": {
                "required": False,
                "prompt": "Where is the object located?",
                "aliases": ["place", "area"]
            }
        },
        "place": {
            "destination": {
                "required": True,
                "prompt": "Where should I place it?",
                "aliases": ["location", "place", "area"]
            }
        },
        "inspect": {
            "target": {
                "required": True,
                "prompt": "What would you like me to inspect?",
                "aliases": ["object", "area", "location"]
            }
        }
    }
    
    def get_required_slots(self, intent: str) -> Dict[str, Any]:
        """Get required slots for an intent."""
        return self.INTENT_SLOTS.get(intent, {})
    
    def check_slots(
        self,
        intent: str,
        entities: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Check if all required slots are filled.
        
        Args:
            intent: Intent to check
            entities: Extracted entities
            
        Returns:
            Tuple of (all_filled, missing_slot_prompt)
        """
        slots = self.get_required_slots(intent)
        
        for slot_name, slot_config in slots.items():
            if not slot_config.get("required", False):
                continue
            
            # Check slot and aliases
            found = slot_name in entities
            if not found:
                for alias in slot_config.get("aliases", []):
                    if alias in entities:
                        found = True
                        break
            
            if not found:
                return False, slot_config.get("prompt", f"What is the {slot_name}?")
        
        return True, None
    
    def normalize_entities(
        self,
        intent: str,
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize entities to canonical slot names.
        
        Args:
            intent: Intent to normalize for
            entities: Extracted entities
            
        Returns:
            Normalized entities dict
        """
        slots = self.get_required_slots(intent)
        normalized = {}
        
        for slot_name, slot_config in slots.items():
            # Check canonical name
            if slot_name in entities:
                normalized[slot_name] = entities[slot_name]
                continue
            
            # Check aliases
            for alias in slot_config.get("aliases", []):
                if alias in entities:
                    normalized[slot_name] = entities[alias]
                    break
        
        return normalized


__all__ = [
    'ResponseType',
    'ContextAwareResponse',
    'ContextAwareResponseGenerator',
    'SlotFillingManager',
]
