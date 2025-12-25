"""
Intent Parser using LLM
Extracts robot commands from natural language
"""

import json
import os
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ActionType(str, Enum):
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    PICK = "pick"
    PLACE = "place"
    STOP = "stop"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass
class RobotIntent:
    """Parsed robot command intent."""
    action: ActionType
    target: Optional[str] = None
    coordinates: Optional[list] = None
    speed: str = "normal"
    confidence: float = 0.0
    raw_text: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


SYSTEM_PROMPT = """You are a robot command parser. Extract intent from natural language.

Return JSON only:
{
    "action": "navigate|inspect|pick|place|stop|status",
    "target": "zone name or object",
    "coordinates": [x, y] or null,
    "speed": "slow|normal|fast",
    "confidence": 0.0-1.0
}

Zone mappings:
- "zone a", "zone 1", "start" → [0, 0]
- "zone b", "zone 2", "loading" → [10, 0]
- "zone c", "zone 3", "storage" → [10, 10]
- "north shelf" → [5, 15]
- "south shelf" → [5, -5]
"""


class IntentParser:
    """Parse natural language to robot intents."""
    
    def __init__(self, provider: str = "nim"):
        self.provider = provider
        self._setup_client()
    
    def _setup_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == "nim":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.getenv("NIM_API_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
                api_key=os.getenv("NIM_API_KEY", ""),
            )
            self.model = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
        elif self.provider == "ollama":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1",
                api_key="ollama",
            )
            self.model = "llama3"
        else:
            self.client = None
    
    def parse(self, text: str) -> RobotIntent:
        """Parse text into robot intent."""
        if not self.client:
            return self._fallback_parse(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return RobotIntent(
                action=ActionType(data.get("action", "unknown")),
                target=data.get("target"),
                coordinates=data.get("coordinates"),
                speed=data.get("speed", "normal"),
                confidence=data.get("confidence", 0.8),
                raw_text=text,
            )
        except Exception as e:
            print(f"LLM error: {e}, using fallback")
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> RobotIntent:
        """Simple rule-based fallback parser."""
        text_lower = text.lower()
        
        # Action detection
        if any(w in text_lower for w in ["stop", "halt", "freeze"]):
            return RobotIntent(ActionType.STOP, confidence=0.9, raw_text=text)
        
        if any(w in text_lower for w in ["status", "where", "position"]):
            return RobotIntent(ActionType.STATUS, confidence=0.9, raw_text=text)
        
        if any(w in text_lower for w in ["inspect", "check", "scan"]):
            action = ActionType.INSPECT
        elif any(w in text_lower for w in ["pick", "grab", "get"]):
            action = ActionType.PICK
        elif any(w in text_lower for w in ["place", "put", "drop"]):
            action = ActionType.PLACE
        elif any(w in text_lower for w in ["move", "go", "navigate"]):
            action = ActionType.NAVIGATE
        else:
            action = ActionType.UNKNOWN
        
        # Target detection
        target = None
        coordinates = None
        
        if "zone b" in text_lower or "zone 2" in text_lower:
            target = "zone_b"
            coordinates = [10.0, 0.0]
        elif "zone a" in text_lower or "zone 1" in text_lower:
            target = "zone_a"
            coordinates = [0.0, 0.0]
        elif "zone c" in text_lower or "zone 3" in text_lower:
            target = "zone_c"
            coordinates = [10.0, 10.0]
        elif "north shelf" in text_lower:
            target = "north_shelf"
            coordinates = [5.0, 15.0]
        elif "south shelf" in text_lower:
            target = "south_shelf"
            coordinates = [5.0, -5.0]
        
        return RobotIntent(
            action=action,
            target=target,
            coordinates=coordinates,
            confidence=0.7,
            raw_text=text,
        )


if __name__ == "__main__":
    parser = IntentParser(provider="fallback")
    
    test_phrases = [
        "Move to zone B",
        "Inspect the north shelf",
        "Stop immediately",
        "Pick up the package",
        "What's your status?",
    ]
    
    for phrase in test_phrases:
        intent = parser.parse(phrase)
        print(f"'{phrase}' → {intent.to_dict()}")
