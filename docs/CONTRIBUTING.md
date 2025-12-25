# Contributing to Digital Twin Robotics Lab

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to:
- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other contributors

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for simulation)
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/digital-twin-robot-NVD.git
cd digital-twin-robot-NVD
git remote add upstream https://github.com/JackAmichai/digital-twin-robot-NVD.git
```

---

## Development Setup

### 1. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 3. Docker Development

```bash
# Build containers with development settings
docker compose -f docker-compose.yml -f docker-compose.dev.yml build

# Start in development mode
docker compose --profile dev up
```

---

## How to Contribute

### Types of Contributions

| Type | Description |
|------|-------------|
| 🐛 **Bug Fixes** | Fix issues reported in GitHub Issues |
| ✨ **Features** | Add new functionality (discuss first in Issues) |
| 📝 **Documentation** | Improve README, add examples, fix typos |
| 🧪 **Tests** | Add test coverage, fix flaky tests |
| 🎨 **Refactoring** | Improve code quality without changing behavior |
| 🔧 **Configuration** | Improve Docker, CI/CD, tooling |

### Workflow

1. **Check Issues**: Look for existing issues or create a new one
2. **Discuss**: For major changes, open an issue first to discuss
3. **Branch**: Create a feature branch from `main`
4. **Develop**: Make your changes with tests
5. **Test**: Run the test suite locally
6. **Submit**: Open a pull request

---

## Pull Request Process

### Branch Naming

```
feature/add-lidar-filter
bugfix/fix-nav2-timeout
docs/update-setup-guide
refactor/cleanup-intent-parser
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add emergency stop voice command
fix: resolve Nav2 goal cancellation race condition
docs: add Foxglove configuration guide
test: add integration tests for Redis bridge
refactor: extract zone coordinates to config file
```

### PR Checklist

Before submitting, ensure:

- [ ] Code follows the project's coding standards
- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] New features have corresponding tests
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with `main`

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
Describe how you tested the changes.

## Related Issues
Closes #123
```

---

## Coding Standards

### Python

- **Style**: Follow PEP 8
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `ruff` or `flake8`
- **Type Hints**: Use type hints for all functions

```python
# Good
def calculate_distance(point_a: Point, point_b: Point) -> float:
    """Calculate Euclidean distance between two points."""
    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y
    return math.sqrt(dx**2 + dy**2)

# Bad
def calc_dist(a, b):
    return math.sqrt((b.x-a.x)**2 + (b.y-a.y)**2)
```

### ROS 2

- Follow [ROS 2 style guide](https://docs.ros.org/en/humble/Contributing/Code-Style-Language-Versions.html)
- Use descriptive node and topic names
- Document message types and service interfaces

### Docker

- Use multi-stage builds
- Minimize layer count
- Don't run as root in containers
- Use specific image tags (not `latest`)

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v

# Run specific test
pytest tests/test_integration.py::TestIntentParser::test_fallback_parser_go_to_commands -v
```

### Writing Tests

```python
import pytest
from your_module import YourClass

class TestYourClass:
    """Tests for YourClass."""
    
    def test_basic_functionality(self):
        """Test that basic functionality works."""
        obj = YourClass()
        result = obj.do_something()
        assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        obj = YourClass()
        result = await obj.async_method()
        assert result is not None
    
    @pytest.fixture
    def mock_redis(self):
        """Provide a mock Redis client."""
        from unittest.mock import AsyncMock
        return AsyncMock()
```

### Test Categories

| Marker | Purpose | Run Command |
|--------|---------|-------------|
| `@pytest.mark.unit` | Unit tests | `pytest -m unit` |
| `@pytest.mark.integration` | Integration tests | `pytest -m integration` |
| `@pytest.mark.slow` | Slow tests | `pytest -m slow` |
| `@pytest.mark.gpu` | Requires GPU | `pytest -m gpu` |

---

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def navigate_to_zone(zone_name: str, speed: float = 0.5) -> NavigationResult:
    """Navigate robot to specified zone.
    
    Args:
        zone_name: Name of target zone (e.g., 'loading_dock').
        speed: Maximum navigation speed in m/s.
    
    Returns:
        NavigationResult containing success status and metrics.
    
    Raises:
        UnknownZoneError: If zone_name is not recognized.
        NavigationError: If path planning fails.
    
    Example:
        >>> result = navigate_to_zone('storage', speed=0.3)
        >>> print(result.distance)
        5.2
    """
```

### Markdown Files

- Use proper heading hierarchy
- Include code examples where helpful
- Add table of contents for long documents
- Use Mermaid for diagrams

---

## Questions?

- **Bugs**: Open a [GitHub Issue](https://github.com/JackAmichai/digital-twin-robot-NVD/issues)
- **Features**: Start a [Discussion](https://github.com/JackAmichai/digital-twin-robot-NVD/discussions)
- **Security**: Email the maintainers directly

---

Thank you for contributing! 🤖
