# SDK Generation Module

Generate client SDKs from OpenAPI specifications.

## Features

- **Multi-Language**: Python, TypeScript, Go, Java, C#
- **OpenAPI 3.x**: Full spec support
- **Async Support**: Async/await clients
- **Type Safety**: Generated type definitions

## Supported Languages

| Language | Status | Features |
|----------|--------|----------|
| Python | ✓ Full | Sync + async, models, httpx |
| TypeScript | ✓ Full | Types, fetch-based |
| Go | Planned | - |
| Java | Planned | - |
| C# | Planned | - |

## Usage

### Quick Generation
```python
from sdk import generate_python_sdk, generate_typescript_sdk

# Python SDK
generate_python_sdk(
    openapi_path="api/openapi/robotics-api.yaml",
    output_dir="./sdks/python",
    package_name="robotics_client",
)

# TypeScript SDK
generate_typescript_sdk(
    openapi_path="api/openapi/robotics-api.yaml",
    output_dir="./sdks/typescript",
    package_name="@robotics/client",
)
```

### Custom Configuration
```python
from sdk import SDKGenerator, SDKConfig, Language

generator = SDKGenerator("api/openapi/robotics-api.yaml")

config = SDKConfig(
    language=Language.PYTHON,
    package_name="my_robotics_sdk",
    version="2.0.0",
    output_dir="./custom_sdk",
    base_url="https://api.robotics.example.com",
    include_models=True,
    include_async=True,
    author="Your Name",
    license="Apache-2.0",
)

output_path = generator.generate(config)
```

## Generated Python SDK

### Installation
```bash
cd generated_sdk
pip install -e .
```

### Usage
```python
from robotics_sdk import RoboticsClient

client = RoboticsClient(
    base_url="https://api.robotics.local",
    api_key="your-api-key",
)

# Sync call
robots = client.list_robots()

# Async call
import asyncio
robots = asyncio.run(client.list_robots_async())

# Clean up
client.close()
```

## Generated TypeScript SDK

### Installation
```bash
cd generated_sdk
npm install
npm run build
```

### Usage
```typescript
import { RoboticsClient } from 'robotics-sdk';

const client = new RoboticsClient({
  baseUrl: 'https://api.robotics.local',
  apiKey: 'your-api-key',
});

const robots = await client.listRobots();
```

## Customization

### Extend Generated Client
```python
from robotics_sdk import RoboticsClient

class MyClient(RoboticsClient):
    def custom_method(self):
        # Add custom functionality
        pass
```

## CI/CD Integration

```yaml
# .github/workflows/sdk.yml
- name: Generate SDKs
  run: |
    python -c "
    from sdk import generate_python_sdk
    generate_python_sdk('api/openapi/robotics-api.yaml')
    "
```
