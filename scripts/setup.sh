#!/bin/bash
# =============================================================================
# Initial Setup Script
# =============================================================================

set -e

echo "🚀 Digital Twin Robotics Lab - Initial Setup"
echo "============================================="

# Create directories
echo "📁 Creating directories..."
mkdir -p data/logs data/redis data/recordings
mkdir -p cognitive_service/src cognitive_service/config
mkdir -p simulation/environments simulation/robots simulation/scripts
mkdir -p config docs/diagrams

# Create .gitkeep files
touch data/.gitkeep
touch data/logs/.gitkeep
touch cognitive_service/src/.gitkeep
touch simulation/environments/.gitkeep

# Setup .env if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Edit .env with your API keys!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your NVIDIA API keys"
echo "  2. Run: make check-env"
echo "  3. Run: make build"
echo "  4. Run: make up"
