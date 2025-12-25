#!/bin/bash
# =============================================================================
# Digital Twin Robotics Lab - Demo Launcher
# =============================================================================
# Launches the full voice-controlled navigation demo
# =============================================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🤖 Digital Twin Robotics Lab - Demo Mode                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running in docker context
if [ -f /.dockerenv ]; then
    echo -e "${GREEN}Running inside Docker container${NC}"
else
    echo -e "${YELLOW}Starting via Docker Compose...${NC}"
    
    # Start all services
    echo "Starting services..."
    docker compose --profile full up -d
    
    # Wait for services to be healthy
    echo "Waiting for services to initialize..."
    sleep 10
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Demo is running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Available commands (speak or type):"
echo "  • 'Move to Zone B'"
echo "  • 'Inspect the north shelf'"
echo "  • 'Go to the loading dock'"
echo "  • 'Stop'"
echo "  • 'What is your status?'"
echo ""
echo "Visualization:"
echo "  • Foxglove Studio: http://localhost:8080"
echo "  • Isaac Sim Stream: http://localhost:8211"
echo ""
echo "Press Ctrl+C to stop the demo"
echo ""

# If we have a text interface, read commands
if [ -t 0 ]; then
    echo "Type a command (or 'quit' to exit):"
    while true; do
        read -p "> " cmd
        if [ "$cmd" = "quit" ] || [ "$cmd" = "exit" ]; then
            break
        fi
        # Send command via Redis
        docker compose exec redis redis-cli PUBLISH robot_commands "{\"action\":\"navigate\",\"raw_text\":\"$cmd\"}"
    done
fi

echo ""
echo -e "${YELLOW}Stopping demo...${NC}"
docker compose --profile full down
echo -e "${GREEN}Demo stopped.${NC}"
