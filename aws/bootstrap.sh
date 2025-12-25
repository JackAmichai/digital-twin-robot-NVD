#!/bin/bash
# Digital Twin Robotics Lab - EC2 Bootstrap Script
# This script runs on first boot to set up the environment

set -e
exec > >(tee /var/log/user-data.log) 2>&1

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Digital Twin Robotics Lab - Bootstrap Script                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Started at: $(date)"
echo ""

# Configuration (passed from Terraform)
NGC_API_KEY="${ngc_api_key}"
NIM_API_KEY="${nim_api_key}"

# =============================================================================
# System Updates
# =============================================================================
echo ">>> Step 1/8: Updating system packages..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# =============================================================================
# Install Docker
# =============================================================================
echo ">>> Step 2/8: Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu
fi

# =============================================================================
# Install Docker Compose
# =============================================================================
echo ">>> Step 3/8: Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    curl -L "https://github.com/docker/compose/releases/download/$${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# =============================================================================
# Install NVIDIA Container Toolkit
# =============================================================================
echo ">>> Step 4/8: Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# =============================================================================
# Install NVIDIA Drivers (if not present)
# =============================================================================
echo ">>> Step 5/8: Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    apt-get install -y nvidia-driver-535
    echo "NOTE: Reboot may be required for drivers to load"
fi

# Verify GPU
echo "GPU Status:"
nvidia-smi || echo "WARNING: nvidia-smi not available yet (may need reboot)"

# =============================================================================
# Install Development Tools
# =============================================================================
echo ">>> Step 6/8: Installing development tools..."
apt-get install -y \
    git \
    htop \
    nvtop \
    tmux \
    jq \
    tree \
    curl \
    wget \
    unzip \
    python3-pip \
    python3-venv

# =============================================================================
# Clone Repository
# =============================================================================
echo ">>> Step 7/8: Cloning repository..."
cd /home/ubuntu

if [ ! -d "digital-twin-robot-NVD" ]; then
    git clone https://github.com/JackAmichai/digital-twin-robot-NVD.git
fi

chown -R ubuntu:ubuntu digital-twin-robot-NVD

# =============================================================================
# Configure Environment
# =============================================================================
echo ">>> Step 8/8: Configuring environment..."
cd digital-twin-robot-NVD

# Create .env file
cat > .env << ENVFILE
# ============================================
# Digital Twin Robotics Lab - Configuration
# ============================================

# NVIDIA API Keys (REQUIRED - get from build.nvidia.com)
NGC_API_KEY=$${NGC_API_KEY}
NIM_API_KEY=$${NIM_API_KEY}

# ROS 2 Configuration
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Service URLs (internal)
RIVA_SERVER=riva:50051
NIM_URL=http://nim:8000/v1

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Isaac Sim Configuration
DISPLAY=:0
ACCEPT_EULA=Y

# Foxglove Configuration
FOXGLOVE_PORT=8765
ENVFILE

chown ubuntu:ubuntu .env

# Create systemd service for optional auto-start
cat > /etc/systemd/system/digital-twin.service << 'SERVICE'
[Unit]
Description=Digital Twin Robotics Lab
After=docker.service nvidia-persistenced.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/digital-twin-robot-NVD
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ubuntu
Group=ubuntu
Environment=HOME=/home/ubuntu

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
# Note: Not enabling auto-start by default - user should start manually

# =============================================================================
# NGC Login (if API key provided)
# =============================================================================
if [ -n "$NGC_API_KEY" ]; then
    echo "Logging into NGC..."
    docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY" || true
fi

# =============================================================================
# Create helper scripts
# =============================================================================
cat > /home/ubuntu/start-digital-twin.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/digital-twin-robot-NVD
docker-compose up -d
echo ""
echo "Digital Twin Robotics Lab started!"
echo "Access Foxglove at: http://$(curl -s ifconfig.me):8080"
echo "Access Isaac Sim at: http://$(curl -s ifconfig.me):8211"
SCRIPT
chmod +x /home/ubuntu/start-digital-twin.sh
chown ubuntu:ubuntu /home/ubuntu/start-digital-twin.sh

cat > /home/ubuntu/stop-digital-twin.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/digital-twin-robot-NVD
docker-compose down
echo "Digital Twin Robotics Lab stopped."
SCRIPT
chmod +x /home/ubuntu/stop-digital-twin.sh
chown ubuntu:ubuntu /home/ubuntu/stop-digital-twin.sh

cat > /home/ubuntu/status-digital-twin.sh << 'SCRIPT'
#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi
echo ""
echo "=== Docker Containers ==="
docker ps
echo ""
echo "=== Public IP ==="
curl -s ifconfig.me
echo ""
SCRIPT
chmod +x /home/ubuntu/status-digital-twin.sh
chown ubuntu:ubuntu /home/ubuntu/status-digital-twin.sh

# =============================================================================
# Mark setup complete
# =============================================================================
touch /home/ubuntu/.setup-complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE!                               ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  Next steps:                                                     ║"
echo "║  1. Edit .env file with your NVIDIA API keys:                    ║"
echo "║     cd digital-twin-robot-NVD && nano .env                       ║"
echo "║                                                                  ║"
echo "║  2. Start the platform:                                          ║"
echo "║     ./start-digital-twin.sh                                      ║"
echo "║     OR: cd digital-twin-robot-NVD && docker-compose up -d        ║"
echo "║                                                                  ║"
echo "║  Helper scripts available:                                       ║"
echo "║  - ~/start-digital-twin.sh   - Start all services                ║"
echo "║  - ~/stop-digital-twin.sh    - Stop all services                 ║"
echo "║  - ~/status-digital-twin.sh  - Check system status               ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Completed at: $(date)"
