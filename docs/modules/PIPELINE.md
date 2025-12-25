# Data Pipeline Module

ETL and streaming data processing framework.

## Features

- **Pipeline Stages**: Transform, filter, enrich
- **Streaming Support**: Real-time processing
- **Metrics Tracking**: Throughput, errors
- **Fluent API**: Chainable operations

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| Transform | Apply function to modify data |
| Filter | Remove records based on predicate |
| Enrich | Add computed fields |

## Usage

### Basic ETL Pipeline
```python
from pipeline import Pipeline, DataSource, DataSink

class FileSource(DataSource):
    async def read(self):
        for line in open("data.jsonl"):
            yield json.loads(line)

class DatabaseSink(DataSink):
    async def write(self, data):
        await db.insert(data)

pipeline = Pipeline(
    name="robot-telemetry-etl",
    source=FileSource(),
    sink=DatabaseSink(),
)

pipeline.filter(lambda x: x["status"] == "active")
pipeline.transform(lambda x: {**x, "processed_at": datetime.utcnow()})
pipeline.enrich(lambda x: {"region": lookup_region(x["robot_id"])})

metrics = await pipeline.run()
print(f"Processed: {metrics.records_processed}")
print(f"Throughput: {metrics.throughput:.2f} rec/s")
```

### Fluent API
```python
pipeline = (
    Pipeline("etl", source, sink)
    .filter(lambda x: x["value"] > 0)
    .transform(normalize)
    .enrich(add_metadata)
)
```

### Stream Processing
```python
from pipeline import StreamProcessor

processor = StreamProcessor(
    batch_size=100,
    window_seconds=5.0,
)

async def handle_batch(batch):
    await analytics.process(batch)

processor.on_batch(handle_batch)

# Ingest records
for record in incoming_data:
    await processor.ingest(record)
```

## Custom Stages
```python
from pipeline import PipelineStage

class AnomalyDetector(PipelineStage):
    async def process(self, data):
        if detect_anomaly(data):
            data["anomaly"] = True
            await alert_service.notify(data)
        return data

pipeline.add_stage(AnomalyDetector())
```

## Metrics

- `records_processed`: Successfully processed
- `records_filtered`: Filtered out
- `errors`: Processing errors
- `duration_seconds`: Total runtime
- `throughput`: Records per second
