# 🛠️ Setup Guide

## Prerequisites

### Hardware Requirements
- **GPU:** NVIDIA RTX 3080+ (12GB+ VRAM)
- **RAM:** 32GB minimum
- **Storage:** 100GB+ SSD

### Software Requirements
- Ubuntu 22.04 or Windows 11 with WSL2
- NVIDIA Driver 525+
- Docker Engine 24+
- NVIDIA Container Toolkit

---

## Ubuntu 22.04 Installation

### 1. Install NVIDIA Drivers
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### 2. Verify GPU
```bash
nvidia-smi
```

### 3. Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Install NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 5. Test GPU in Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## WSL2 Installation (Windows)

### 1. Enable WSL2
```powershell
wsl --install -d Ubuntu-22.04
```

### 2. Install NVIDIA Driver (Windows side)
Download from: https://www.nvidia.com/Download/index.aspx

### 3. Follow Ubuntu steps 3-5 inside WSL2

---

## Project Setup

```bash
cd digital-twin-robotics-lab
cp .env.example .env
# Edit .env with your API keys
make check-env
make build
```

---

## Verification

```bash
make check-gpu    # Verify GPU access
make up           # Start services
make status       # Check containers
```
