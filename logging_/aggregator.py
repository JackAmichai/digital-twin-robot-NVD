"""Log aggregation and structured logging."""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import asyncio
import aiohttp


class LogBackend(Enum):
    """Supported log aggregation backends."""
    ELASTICSEARCH = "elasticsearch"
    LOKI = "loki"
    CONSOLE = "console"


@dataclass
class LogConfig:
    """Log aggregation configuration."""
    
    backend: LogBackend = LogBackend.CONSOLE
    endpoint: str = "http://localhost:9200"
    index_prefix: str = "robotics-lab"
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    extra_labels: dict[str, str] = field(default_factory=dict)


class StructuredLogger:
    """Structured logging with JSON output."""
    
    def __init__(self, name: str, config: LogConfig | None = None):
        self.name = name
        self.config = config or LogConfig()
        self._logger = logging.getLogger(name)
        self._setup_handler()
    
    def _setup_handler(self) -> None:
        """Setup JSON handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
    
    def _create_record(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create structured log record."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.name,
            "message": message,
            "labels": self.config.extra_labels,
            **kwargs,
        }
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        record = self._create_record("INFO", message, **kwargs)
        self._logger.info(json.dumps(record))
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        record = self._create_record("ERROR", message, **kwargs)
        self._logger.error(json.dumps(record))
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        record = self._create_record("DEBUG", message, **kwargs)
        self._logger.debug(json.dumps(record))
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        record = self._create_record("WARNING", message, **kwargs)
        self._logger.warning(json.dumps(record))


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        return record.getMessage()


class LogAggregator:
    """Async log aggregator for ELK/Loki."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._running = False
    
    async def start(self) -> None:
        """Start background flush task."""
        self._running = True
        asyncio.create_task(self._flush_loop())
    
    async def stop(self) -> None:
        """Stop and flush remaining logs."""
        self._running = False
        await self._flush()
    
    async def log(self, record: dict[str, Any]) -> None:
        """Add log record to buffer."""
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.config.batch_size:
                await self._flush()
    
    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            await asyncio.sleep(self.config.flush_interval_seconds)
            await self._flush()
    
    async def _flush(self) -> None:
        """Flush buffer to backend."""
        async with self._lock:
            if not self._buffer:
                return
            
            logs = self._buffer.copy()
            self._buffer.clear()
        
        if self.config.backend == LogBackend.ELASTICSEARCH:
            await self._send_to_elasticsearch(logs)
        elif self.config.backend == LogBackend.LOKI:
            await self._send_to_loki(logs)
    
    async def _send_to_elasticsearch(
        self,
        logs: list[dict[str, Any]],
    ) -> None:
        """Send logs to Elasticsearch."""
        bulk_body = []
        index = f"{self.config.index_prefix}-{datetime.utcnow():%Y.%m.%d}"
        
        for log in logs:
            bulk_body.append(json.dumps({"index": {"_index": index}}))
            bulk_body.append(json.dumps(log))
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{self.config.endpoint}/_bulk",
                data="\n".join(bulk_body) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
    
    async def _send_to_loki(self, logs: list[dict[str, Any]]) -> None:
        """Send logs to Loki."""
        streams = [{
            "stream": {
                "app": self.config.index_prefix,
                **self.config.extra_labels,
            },
            "values": [
                [str(int(datetime.utcnow().timestamp() * 1e9)), json.dumps(log)]
                for log in logs
            ],
        }]
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{self.config.endpoint}/loki/api/v1/push",
                json={"streams": streams},
                headers={"Content-Type": "application/json"},
            )
