"""Data pipeline with ETL and streaming capabilities."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, TypeVar
from uuid import uuid4


T = TypeVar("T")


class DataSource(ABC):
    """Abstract data source."""
    
    @abstractmethod
    async def read(self) -> AsyncIterator[dict[str, Any]]:
        """Read data from source."""
        pass


class DataSink(ABC):
    """Abstract data sink."""
    
    @abstractmethod
    async def write(self, data: dict[str, Any]) -> None:
        """Write data to sink."""
        pass
    
    async def flush(self) -> None:
        """Flush any buffered data."""
        pass


class PipelineStage(ABC):
    """Abstract pipeline processing stage."""
    
    @abstractmethod
    async def process(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Process data. Return None to filter out."""
        pass


class TransformStage(PipelineStage):
    """Apply transformation function."""
    
    def __init__(self, transform_fn: Callable[[dict], dict]):
        self.transform_fn = transform_fn
    
    async def process(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.transform_fn(data)


class FilterStage(PipelineStage):
    """Filter data based on predicate."""
    
    def __init__(self, predicate: Callable[[dict], bool]):
        self.predicate = predicate
    
    async def process(self, data: dict[str, Any]) -> dict[str, Any] | None:
        return data if self.predicate(data) else None


class EnrichStage(PipelineStage):
    """Enrich data with additional fields."""
    
    def __init__(self, enrich_fn: Callable[[dict], dict[str, Any]]):
        self.enrich_fn = enrich_fn
    
    async def process(self, data: dict[str, Any]) -> dict[str, Any]:
        enrichment = self.enrich_fn(data)
        return {**data, **enrichment}


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    
    records_processed: int = 0
    records_filtered: int = 0
    errors: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def throughput(self) -> float:
        if self.duration_seconds > 0:
            return self.records_processed / self.duration_seconds
        return 0.0


class Pipeline:
    """ETL data pipeline with stages."""
    
    def __init__(
        self,
        name: str,
        source: DataSource,
        sink: DataSink,
    ):
        self.name = name
        self.id = str(uuid4())
        self.source = source
        self.sink = sink
        self._stages: list[PipelineStage] = []
        self.metrics = PipelineMetrics()
    
    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Add processing stage."""
        self._stages.append(stage)
        return self
    
    def transform(self, fn: Callable[[dict], dict]) -> "Pipeline":
        """Add transform stage."""
        return self.add_stage(TransformStage(fn))
    
    def filter(self, predicate: Callable[[dict], bool]) -> "Pipeline":
        """Add filter stage."""
        return self.add_stage(FilterStage(predicate))
    
    def enrich(self, fn: Callable[[dict], dict]) -> "Pipeline":
        """Add enrichment stage."""
        return self.add_stage(EnrichStage(fn))
    
    async def run(self) -> PipelineMetrics:
        """Execute the pipeline."""
        self.metrics = PipelineMetrics(start_time=datetime.utcnow())
        
        try:
            async for record in self.source.read():
                result = record
                
                for stage in self._stages:
                    if result is None:
                        break
                    try:
                        result = await stage.process(result)
                    except Exception:
                        self.metrics.errors += 1
                        result = None
                
                if result is not None:
                    await self.sink.write(result)
                    self.metrics.records_processed += 1
                else:
                    self.metrics.records_filtered += 1
            
            await self.sink.flush()
        finally:
            self.metrics.end_time = datetime.utcnow()
        
        return self.metrics


class StreamProcessor:
    """Real-time stream processing."""
    
    def __init__(self, batch_size: int = 100, window_seconds: float = 5.0):
        self.batch_size = batch_size
        self.window_seconds = window_seconds
        self._buffer: list[dict[str, Any]] = []
        self._handlers: list[Callable] = []
    
    def on_batch(self, handler: Callable[[list[dict]], None]) -> None:
        """Register batch handler."""
        self._handlers.append(handler)
    
    async def ingest(self, data: dict[str, Any]) -> None:
        """Ingest single record."""
        self._buffer.append(data)
        
        if len(self._buffer) >= self.batch_size:
            await self._flush_batch()
    
    async def _flush_batch(self) -> None:
        """Flush current batch to handlers."""
        if not self._buffer:
            return
        
        batch = self._buffer.copy()
        self._buffer.clear()
        
        for handler in self._handlers:
            await handler(batch)
    
    async def start_windowing(self) -> None:
        """Start time-based windowing."""
        while True:
            await asyncio.sleep(self.window_seconds)
            await self._flush_batch()
