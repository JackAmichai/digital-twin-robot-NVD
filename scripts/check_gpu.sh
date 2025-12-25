#!/bin/bash
# =============================================================================
# GPU Check Script
# =============================================================================

echo "🎮 GPU Configuration Check"
echo "=========================="
echo ""

if ! command -v nvidia-smi &>/dev/null; then
    echo "❌ nvidia-smi not found. Install NVIDIA drivers."
    exit 1
fi

echo "📊 GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "🐳 Docker GPU Test:"
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi -L

echo ""
echo "✅ GPU ready for Digital Twin Robotics Lab!"
