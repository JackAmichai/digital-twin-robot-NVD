# Digital Twin Robotics Lab - User Setup Guide

> **Complete step-by-step guide to deploy and run the voice-controlled robotics simulation platform**

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Get NVIDIA API Keys](#2-get-nvidia-api-keys)
3. [Set Up AWS Account](#3-set-up-aws-account)
4. [Deploy Infrastructure](#4-deploy-infrastructure)
5. [Configure the Instance](#5-configure-the-instance)
6. [Run the Platform](#6-run-the-platform)
7. [Access the Services](#7-access-the-services)
8. [Test Voice Commands](#8-test-voice-commands)
9. [Cost Management](#9-cost-management)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Required Accounts

- [ ] **AWS Account** with billing enabled
- [ ] **NVIDIA Developer Account** (free) - [Sign up here](https://developer.nvidia.com/developer-program)
- [ ] **GitHub Account** (optional, for code access)

### Local Tools (Install on Your Computer)

- [ ] **AWS CLI v2** - [Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  ```powershell
  # Windows (PowerShell as Admin)
  msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
  ```

- [ ] **Terraform** (Option A) - [Download](https://www.terraform.io/downloads)
  ```powershell
  # Windows (using Chocolatey)
  choco install terraform
  ```

- [ ] **SSH Client** - Built into Windows 10+ (PowerShell) or use [PuTTY](https://www.putty.org/)

### Estimated Costs

| Instance Type | GPU | VRAM | Cost/Hour | Cost/Month (24/7) |
|--------------|-----|------|-----------|-------------------|
| g4dn.xlarge | T4 | 16GB | $0.53 | ~$380 |
| **g5.xlarge** | **A10G** | **24GB** | **$1.01** | **~$725** |
| g5.2xlarge | A10G | 24GB | $1.21 | ~$875 |

> 💡 **Tip**: Use g5.xlarge for best performance. Stop instance when not in use!

---

## 2. Get NVIDIA API Keys

### Step 2.1: Get NGC API Key

1. Go to [NVIDIA NGC](https://ngc.nvidia.com/)
2. Sign in or create account
3. Click your profile icon (top right) → **Setup**
4. Click **Generate API Key**
5. Copy and save the key securely

```
Your NGC API Key: nvapi-xxxx-xxxx-xxxx
```

### Step 2.2: Get NIM API Key

1. Go to [NVIDIA Build](https://build.nvidia.com/)
2. Sign in with your NVIDIA account
3. Search for "Llama 3.1 8B Instruct"
4. Click **Get API Key**
5. Copy and save the key securely

```
Your NIM API Key: nvapi-xxxx-xxxx-xxxx
```

> ⚠️ **Important**: Keep these keys secure. Never commit them to git!

---

## 3. Set Up AWS Account

### Step 3.1: Create EC2 Key Pair

1. Go to [AWS Console → EC2 → Key Pairs](https://console.aws.amazon.com/ec2/home#KeyPairs:)
2. Click **Create key pair**
3. Settings:
   - Name: `digital-twin-key`
   - Key pair type: RSA
   - Private key file format: `.pem`
4. Click **Create key pair**
5. Save the downloaded `.pem` file

```powershell
# Move the key to your .ssh folder
mkdir -p $HOME\.ssh
move $HOME\Downloads\digital-twin-key.pem $HOME\.ssh\

# Set proper permissions (PowerShell)
icacls "$HOME\.ssh\digital-twin-key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

### Step 3.2: Configure AWS CLI

```powershell
aws configure
```

Enter:
- AWS Access Key ID: `<your-access-key>`
- AWS Secret Access Key: `<your-secret-key>`
- Default region: `us-east-1` (or your preferred region)
- Output format: `json`

### Step 3.3: Request GPU Instance Quota (if needed)

1. Go to [Service Quotas → EC2](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas)
2. Search for "Running On-Demand G and VT instances"
3. If quota is 0, click **Request quota increase**
4. Request at least **8 vCPUs** (g5.xlarge needs 4)

> ⏱️ Quota increases typically take 15-30 minutes

---

## 4. Deploy Infrastructure

Choose **Option A (Terraform)** or **Option B (CloudFormation)**:

### Option A: Deploy with Terraform (Recommended)

#### Step 4A.1: Clone Repository

```powershell
git clone https://github.com/JackAmichai/digital-twin-robot-NVD.git
cd digital-twin-robot-NVD\aws
```

#### Step 4A.2: Create Configuration

```powershell
# Copy example config
copy terraform.tfvars.example terraform.tfvars

# Edit the file
notepad terraform.tfvars
```

Update these values:
```hcl
aws_region       = "us-east-1"
key_pair_name    = "digital-twin-key"
allowed_ssh_cidr = "YOUR.IP.ADDRESS/32"  # Get from whatismyip.com
ngc_api_key      = "nvapi-xxxx"          # From step 2.1
nim_api_key      = "nvapi-xxxx"          # From step 2.2
```

#### Step 4A.3: Deploy

```powershell
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy (type 'yes' when prompted)
terraform apply
```

#### Step 4A.4: Save Outputs

```powershell
# Show outputs
terraform output

# Save SSH command
terraform output ssh_command
```

---

### Option B: Deploy with CloudFormation

#### Step 4B.1: Deploy Stack

```powershell
# Navigate to aws folder
cd digital-twin-robot-NVD\aws

# Create stack
aws cloudformation create-stack `
  --stack-name digital-twin-robotics `
  --template-body file://cloudformation.yaml `
  --parameters `
    ParameterKey=KeyPairName,ParameterValue=digital-twin-key `
    ParameterKey=NGCApiKey,ParameterValue=your-ngc-api-key `
    ParameterKey=NIMApiKey,ParameterValue=your-nim-api-key `
  --capabilities CAPABILITY_NAMED_IAM
```

#### Step 4B.2: Wait for Completion

```powershell
# Watch status (wait for CREATE_COMPLETE)
aws cloudformation describe-stacks --stack-name digital-twin-robotics --query "Stacks[0].StackStatus"

# Get outputs
aws cloudformation describe-stacks --stack-name digital-twin-robotics --query "Stacks[0].Outputs"
```

---

## 5. Configure the Instance

### Step 5.1: Connect via SSH

```powershell
# Get the public IP from Terraform or CloudFormation outputs
$IP = "YOUR_INSTANCE_IP"

# Connect
ssh -i $HOME\.ssh\digital-twin-key.pem ubuntu@$IP
```

### Step 5.2: Wait for Setup to Complete

```bash
# Watch the setup log
tail -f /var/log/user-data.log

# Wait until you see "SETUP COMPLETE!"
# Press Ctrl+C to exit
```

### Step 5.3: Verify GPU

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx    Driver Version: 535.xxx    CUDA Version: 12.x         |
|-------------------------------+----------------------+----------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         Off  | 00000000:00:1E.0 Off |                    0 |
|  0%   25C    P0    51W / 300W |      0MiB / 23028MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Step 5.4: Configure API Keys (if not done during deployment)

```bash
cd ~/digital-twin-robot-NVD
nano .env
```

Update these lines:
```bash
NGC_API_KEY=nvapi-xxxx-your-key
NIM_API_KEY=nvapi-xxxx-your-key
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

### Step 5.5: Login to NGC (for Docker images)

```bash
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
```

---

## 6. Run the Platform

### Step 6.1: Start All Services

```bash
# Using helper script
~/start-digital-twin.sh

# OR manually
cd ~/digital-twin-robot-NVD
docker-compose up -d
```

### Step 6.2: Monitor Startup

```bash
# Watch logs
docker-compose logs -f

# Check container status
docker ps
```

Expected containers:
- `cognitive_service` - Voice processing + LLM
- `ros2_navigation` - ROS 2 with Nav2
- `isaac_sim` - NVIDIA Isaac Sim
- `redis` - Message broker
- `foxglove_bridge` - Visualization

### Step 6.3: Verify Services

```bash
# Run health check
~/status-digital-twin.sh

# Or check individually
docker-compose ps
```

---

## 7. Access the Services

### From Your Local Computer

| Service | URL | Description |
|---------|-----|-------------|
| **Foxglove Studio** | `http://YOUR_IP:8080` | ROS 2 visualization |
| **Isaac Sim Livestream** | `http://YOUR_IP:8211` | 3D simulation view |
| **Foxglove Bridge** | `ws://YOUR_IP:8765` | Direct WebSocket |

### Step 7.1: Open Foxglove Studio

1. Open browser: `http://YOUR_IP:8080`
2. Or use Foxglove Desktop app:
   - Download from [foxglove.dev](https://foxglove.dev/download)
   - Connect to `ws://YOUR_IP:8765`

### Step 7.2: View Isaac Sim

1. Open browser: `http://YOUR_IP:8211`
2. Click **Connect** to start streaming
3. Navigate with mouse:
   - Left drag: Rotate view
   - Right drag: Pan
   - Scroll: Zoom

---

## 8. Test Voice Commands

### Step 8.1: Run Demo

```bash
# SSH into instance
ssh -i $HOME\.ssh\digital-twin-key.pem ubuntu@$IP

# Run interactive demo
cd ~/digital-twin-robot-NVD
docker-compose exec cognitive_service python demo.py
```

### Step 8.2: Test Commands

Try these voice commands:
```
"Go to the charging station"
"Navigate to warehouse zone A"
"Move forward two meters"
"Turn left 90 degrees"
"Stop"
"Come back to start"
```

### Step 8.3: Monitor in Foxglove

While running commands:
1. Watch the **3D Panel** for robot movement
2. Check **Robot Path** visualization
3. Monitor **Topic List** for `/cmd_vel`, `/nav2/goal`

---

## 9. Cost Management

### ⚠️ IMPORTANT: Stop Instance When Not in Use!

### Stop Instance (Keep Data)

```powershell
# From your local machine
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID

# Or from AWS Console:
# EC2 → Instances → Select instance → Instance state → Stop
```

### Start Instance

```powershell
aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID
```

> 📝 **Note**: Public IP changes after stop/start. Use `terraform output` to get new IP.

### Destroy Everything (Delete All Resources)

```powershell
# Terraform
cd digital-twin-robot-NVD\aws
terraform destroy

# CloudFormation
aws cloudformation delete-stack --stack-name digital-twin-robotics
```

### Cost Monitoring

1. Go to [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/home)
2. Set up billing alerts in [Budgets](https://console.aws.amazon.com/billing/home#/budgets)

---

## 10. Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, reboot
sudo reboot

# Reconnect and check again
nvidia-smi
```

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker ubuntu

# Logout and login again
exit
# Reconnect via SSH
```

### Container Won't Start

```bash
# Check logs
docker-compose logs cognitive_service
docker-compose logs isaac_sim

# Restart containers
docker-compose down
docker-compose up -d
```

### Cannot Connect to Services

1. Check security group allows your IP
2. Verify instance is running
3. Check firewall:
   ```bash
   sudo ufw status
   ```

### Isaac Sim Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Reduce scene complexity in robot_config.yaml
# Or upgrade to larger instance (g5.2xlarge)
```

### SSH Connection Refused

```powershell
# Check instance is running
aws ec2 describe-instances --instance-ids YOUR_INSTANCE_ID --query "Reservations[0].Instances[0].State.Name"

# Verify security group
aws ec2 describe-security-groups --group-ids YOUR_SG_ID
```

---

## 📞 Support

- **GitHub Issues**: [Report bugs](https://github.com/JackAmichai/digital-twin-robot-NVD/issues)
- **NVIDIA Forums**: [Isaac Sim Support](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-sim/)
- **ROS 2 Answers**: [Nav2 Questions](https://robotics.stackexchange.com/questions/tagged/nav2)

---

## ✅ Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK COMMANDS                           │
├─────────────────────────────────────────────────────────────┤
│ SSH Connect:     ssh -i key.pem ubuntu@IP                   │
│ Start Platform:  ~/start-digital-twin.sh                    │
│ Stop Platform:   ~/stop-digital-twin.sh                     │
│ Check Status:    ~/status-digital-twin.sh                   │
│ View Logs:       docker-compose logs -f                     │
│ Restart:         docker-compose restart                     │
├─────────────────────────────────────────────────────────────┤
│                    SERVICE URLS                             │
├─────────────────────────────────────────────────────────────┤
│ Foxglove:        http://YOUR_IP:8080                        │
│ Isaac Sim:       http://YOUR_IP:8211                        │
│ WebSocket:       ws://YOUR_IP:8765                          │
├─────────────────────────────────────────────────────────────┤
│                    COST SAVING                              │
├─────────────────────────────────────────────────────────────┤
│ Stop (AWS CLI):  aws ec2 stop-instances --instance-ids ID   │
│ Start:           aws ec2 start-instances --instance-ids ID  │
│ Destroy:         terraform destroy                          │
└─────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: 2024 | **Version**: 1.0
