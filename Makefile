# =============================================================================
# Digital Twin Robotics Lab - Makefile
# =============================================================================
# Quick commands for development and deployment
# Usage: make <command>
# =============================================================================

.PHONY: help build up down logs shell clean test check-env

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║     🤖 Digital Twin Robotics Lab - Command Reference        ║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# ENVIRONMENT CHECKS
# =============================================================================
check-env: ## Verify development environment
	@echo "$(CYAN)🔍 Checking development environment...$(RESET)"
	@bash scripts/check_environment.sh

check-gpu: ## Verify GPU and NVIDIA drivers
	@echo "$(CYAN)🎮 Checking GPU configuration...$(RESET)"
	@bash scripts/check_gpu.sh

# =============================================================================
# BUILD COMMANDS
# =============================================================================
build: ## Build all Docker images
	@echo "$(CYAN)🔨 Building all containers...$(RESET)"
	docker compose build

build-ros2: ## Build only ROS 2 container
	@echo "$(CYAN)🔨 Building ROS 2 container...$(RESET)"
	docker compose build ros2

build-cognitive: ## Build only Cognitive container
	@echo "$(CYAN)🔨 Building Cognitive container...$(RESET)"
	docker compose build cognitive

build-nocache: ## Build all containers without cache
	@echo "$(CYAN)🔨 Building all containers (no cache)...$(RESET)"
	docker compose build --no-cache

# =============================================================================
# RUN COMMANDS
# =============================================================================
up: ## Start all services
	@echo "$(GREEN)🚀 Starting Digital Twin Robotics Lab...$(RESET)"
	docker compose --profile full up -d
	@echo "$(GREEN)✅ All services started!$(RESET)"
	@make status

up-ros2: ## Start only ROS 2 stack
	@echo "$(GREEN)🚀 Starting ROS 2 stack...$(RESET)"
	docker compose --profile ros2 up -d

up-sim: ## Start ROS 2 + Simulation
	@echo "$(GREEN)🚀 Starting ROS 2 + Simulation...$(RESET)"
	docker compose --profile ros2 --profile simulation up -d

up-dev: ## Start with dev tools (Foxglove, Redis)
	@echo "$(GREEN)🚀 Starting with dev tools...$(RESET)"
	docker compose --profile full --profile dev-tools up -d

down: ## Stop all services
	@echo "$(YELLOW)🛑 Stopping all services...$(RESET)"
	docker compose --profile full --profile dev-tools down

down-v: ## Stop all services and remove volumes
	@echo "$(RED)🗑️  Stopping all services and removing volumes...$(RESET)"
	docker compose --profile full --profile dev-tools down -v

restart: ## Restart all services
	@echo "$(YELLOW)🔄 Restarting all services...$(RESET)"
	docker compose --profile full restart

# =============================================================================
# LOGGING & MONITORING
# =============================================================================
logs: ## Show logs from all containers
	docker compose --profile full logs -f

logs-ros2: ## Show logs from ROS 2 container
	docker compose logs -f ros2

logs-sim: ## Show logs from Isaac Sim container
	docker compose logs -f isaac_sim

logs-cognitive: ## Show logs from Cognitive container
	docker compose logs -f cognitive

status: ## Show status of all containers
	@echo "$(CYAN)📊 Container Status:$(RESET)"
	@docker compose --profile full ps

# =============================================================================
# SHELL ACCESS
# =============================================================================
shell-ros2: ## Open shell in ROS 2 container
	@echo "$(CYAN)🐚 Opening shell in ROS 2 container...$(RESET)"
	docker compose exec ros2 bash

shell-sim: ## Open shell in Isaac Sim container
	@echo "$(CYAN)🐚 Opening shell in Isaac Sim container...$(RESET)"
	docker compose exec isaac_sim bash

shell-cognitive: ## Open shell in Cognitive container
	@echo "$(CYAN)🐚 Opening shell in Cognitive container...$(RESET)"
	docker compose exec cognitive bash

# =============================================================================
# ROS 2 COMMANDS
# =============================================================================
ros2-topics: ## List all ROS 2 topics
	docker compose exec ros2 bash -c "source /opt/ros/humble/setup.bash && ros2 topic list"

ros2-nodes: ## List all ROS 2 nodes
	docker compose exec ros2 bash -c "source /opt/ros/humble/setup.bash && ros2 node list"

ros2-build: ## Build ROS 2 workspace
	@echo "$(CYAN)🔨 Building ROS 2 workspace...$(RESET)"
	docker compose exec ros2 bash -c "cd /ros2_ws && colcon build --symlink-install"

ros2-test: ## Run ROS 2 tests
	@echo "$(CYAN)🧪 Running ROS 2 tests...$(RESET)"
	docker compose exec ros2 bash -c "cd /ros2_ws && colcon test"

# =============================================================================
# DEMO COMMANDS
# =============================================================================
demo: ## Run the main demo scenario
	@echo "$(GREEN)🎬 Starting Demo Mode...$(RESET)"
	@bash scripts/run_demo.sh

demo-nav: ## Run navigation demo
	@echo "$(GREEN)🎬 Starting Navigation Demo...$(RESET)"
	docker compose exec ros2 bash -c "source /ros2_ws/install/setup.bash && ros2 launch robot_bringup navigation_demo.launch.py"

demo-voice: ## Run voice control demo
	@echo "$(GREEN)🎬 Starting Voice Control Demo...$(RESET)"
	@bash scripts/run_voice_demo.sh

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "$(CYAN)🧪 Running all tests...$(RESET)"
	@bash scripts/run_tests.sh

test-integration: ## Run integration tests
	@echo "$(CYAN)🧪 Running integration tests...$(RESET)"
	@bash scripts/run_integration_tests.sh

lint: ## Run linters
	@echo "$(CYAN)🔍 Running linters...$(RESET)"
	docker compose exec ros2 bash -c "cd /ros2_ws && ament_lint"

# =============================================================================
# CLEANUP
# =============================================================================
clean: ## Clean build artifacts
	@echo "$(YELLOW)🧹 Cleaning build artifacts...$(RESET)"
	rm -rf ros2_ws/build ros2_ws/install ros2_ws/log
	@echo "$(GREEN)✅ Clean complete!$(RESET)"

clean-all: down-v clean ## Stop containers, remove volumes, and clean artifacts
	@echo "$(GREEN)✅ Full cleanup complete!$(RESET)"

prune: ## Remove unused Docker resources
	@echo "$(RED)🗑️  Pruning Docker resources...$(RESET)"
	docker system prune -f
	docker volume prune -f

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================
setup: ## Initial project setup
	@echo "$(CYAN)🔧 Running initial setup...$(RESET)"
	@bash scripts/setup.sh

pull: ## Pull all Docker images
	@echo "$(CYAN)📥 Pulling Docker images...$(RESET)"
	docker compose --profile full pull

update: pull build ## Update images and rebuild
	@echo "$(GREEN)✅ Update complete!$(RESET)"

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs: ## Generate documentation
	@echo "$(CYAN)📚 Generating documentation...$(RESET)"
	@bash scripts/generate_docs.sh

# =============================================================================
# QUICK ACCESS
# =============================================================================
foxglove: ## Open Foxglove Studio in browser
	@echo "$(CYAN)🌐 Opening Foxglove Studio...$(RESET)"
	@python -m webbrowser "http://localhost:8080"

isaac: ## Open Isaac Sim streaming viewer
	@echo "$(CYAN)🌐 Opening Isaac Sim Streaming...$(RESET)"
	@python -m webbrowser "http://localhost:8211"
