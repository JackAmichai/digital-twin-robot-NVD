#!/bin/bash
# =============================================================================
# Environment Check Script
# =============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Checking development environment..."
echo ""

PASS=0
FAIL=0

check() {
    if $2 &>/dev/null; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

check "Docker installed" "docker --version"
check "Docker Compose installed" "docker compose version"
check "NVIDIA driver loaded" "nvidia-smi"
check "NVIDIA Container Toolkit" "docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
check "Git installed" "git --version"

echo ""
echo "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"

if [ $FAIL -gt 0 ]; then
    echo -e "${YELLOW}See docs/SETUP.md for installation instructions${NC}"
    exit 1
fi

echo -e "${GREEN}Environment ready!${NC}"
