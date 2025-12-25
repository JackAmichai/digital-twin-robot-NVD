# =============================================================================
# Cognitive Service - Digital Twin Robotics Lab
# =============================================================================
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONPATH=/app/src

CMD ["python", "src/command_bridge.py"]
