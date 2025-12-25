# AWS Deployment Guide

> Technical documentation for deploying Digital Twin Robotics Lab on AWS

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     VPC (10.0.0.0/16)                     │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              Public Subnet (10.0.1.0/24)            │  │  │
│  │  │  ┌───────────────────────────────────────────────┐  │  │  │
│  │  │  │         EC2 g5.xlarge (A10G GPU)              │  │  │  │
│  │  │  │  ┌─────────────────────────────────────────┐  │  │  │  │
│  │  │  │  │           Docker Compose                │  │  │  │  │
│  │  │  │  │  ┌─────────┐ ┌─────────┐ ┌───────────┐  │  │  │  │  │
│  │  │  │  │  │Cognitive│ │  ROS 2  │ │ Isaac Sim │  │  │  │  │  │
│  │  │  │  │  │ Service │ │  Nav2   │ │    4.2    │  │  │  │  │  │
│  │  │  │  │  └────┬────┘ └────┬────┘ └─────┬─────┘  │  │  │  │  │
│  │  │  │  │       └──────┬────┴─────────────┘       │  │  │  │  │
│  │  │  │  │              │ Redis                     │  │  │  │  │
│  │  │  │  └──────────────┴───────────────────────────┘  │  │  │  │
│  │  │  │              Elastic IP                        │  │  │  │
│  │  │  └───────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Instance Requirements

### GPU Specifications

| Instance | GPU | VRAM | vCPUs | RAM | Network | Cost/hr |
|----------|-----|------|-------|-----|---------|---------|
| g4dn.xlarge | T4 | 16GB | 4 | 16GB | 25 Gbps | $0.526 |
| g4dn.2xlarge | T4 | 16GB | 8 | 32GB | 25 Gbps | $0.752 |
| **g5.xlarge** | **A10G** | **24GB** | **4** | **16GB** | **25 Gbps** | **$1.006** |
| g5.2xlarge | A10G | 24GB | 8 | 32GB | 25 Gbps | $1.212 |
| p3.2xlarge | V100 | 16GB | 8 | 61GB | 10 Gbps | $3.060 |

**Recommendation**: `g5.xlarge` provides the best balance of performance and cost.

### Storage Requirements

- **Minimum**: 100GB (base system + containers)
- **Recommended**: 200GB (room for models, logs, sim assets)
- **Type**: gp3 (3000 IOPS, 125 MB/s throughput)

### Memory Breakdown

| Component | GPU Memory | System RAM |
|-----------|------------|------------|
| Isaac Sim | 8-12 GB | 4-6 GB |
| Riva ASR | 2-4 GB | 2 GB |
| NIM LLM | 6-8 GB | 4 GB |
| ROS 2/Nav2 | N/A | 2 GB |
| **Total** | **16-24 GB** | **12-14 GB** |

## Network Configuration

### Security Group Rules

| Port | Protocol | Source | Description |
|------|----------|--------|-------------|
| 22 | TCP | Your IP | SSH access |
| 8080 | TCP | 0.0.0.0/0 | Foxglove Studio |
| 8765 | TCP | 0.0.0.0/0 | Foxglove Bridge |
| 8211 | TCP | 0.0.0.0/0 | Isaac Sim Livestream |
| 49100 | TCP | 0.0.0.0/0 | Isaac Sim WebRTC |
| 7400-7500 | UDP | Your IP | ROS 2 DDS |

### Internal Container Network

```yaml
services:
  cognitive:    # Port 8001 (internal)
  riva:         # Port 50051 (internal)
  nim:          # Port 8000 (internal)
  redis:        # Port 6379 (internal)
  ros2:         # DDS multicast
  isaac_sim:    # Port 8211 (exposed)
  foxglove:     # Port 8765 (exposed)
```

## Deployment Options

### Option 1: Terraform (Recommended)

```bash
cd aws/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init
terraform plan
terraform apply
```

**Benefits**:
- State management
- Easy updates with `terraform apply`
- Modular and extensible

### Option 2: CloudFormation

```bash
aws cloudformation create-stack \
  --stack-name digital-twin-robotics \
  --template-body file://cloudformation.yaml \
  --parameters \
    ParameterKey=KeyPairName,ParameterValue=your-key \
  --capabilities CAPABILITY_NAMED_IAM
```

**Benefits**:
- Native AWS service
- No additional tools needed
- AWS Console visibility

## IAM Permissions

### Required Permissions for Deployment

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "iam:CreateRole",
        "iam:CreateInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:AttachRolePolicy",
        "iam:PassRole",
        "cloudformation:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### EC2 Instance Role

The instance uses these managed policies:
- `AmazonSSMManagedInstanceCore` - AWS Systems Manager access
- `CloudWatchAgentServerPolicy` - CloudWatch logging

## Bootstrap Process

The EC2 instance runs the following setup automatically:

1. **System Updates** - apt-get update/upgrade
2. **Docker Installation** - Docker Engine + Compose
3. **NVIDIA Container Toolkit** - GPU passthrough
4. **NVIDIA Drivers** - nvidia-driver-535
5. **Repository Clone** - From GitHub
6. **Environment Setup** - .env file creation
7. **Helper Scripts** - start/stop/status scripts

### Checking Bootstrap Status

```bash
# SSH into instance
ssh -i key.pem ubuntu@$IP

# Watch setup progress
tail -f /var/log/user-data.log

# Check if complete
ls -la ~/.setup-complete
```

## Performance Tuning

### GPU Optimization

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Set GPU to maximum performance
sudo nvidia-smi -pl 300  # For A10G
```

### Docker Configuration

```bash
# Increase shared memory for CUDA
# Already configured in docker-compose.yml:
# shm_size: '16gb'
```

### Network Tuning

```bash
# Increase UDP buffer sizes for ROS 2
sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.rmem_default=2147483647
```

## Monitoring

### CloudWatch Metrics

- CPU Utilization
- GPU Utilization (custom metric)
- Network I/O
- Disk I/O

### Custom GPU Metrics Script

```bash
#!/bin/bash
# /opt/gpu-metrics.sh

UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

aws cloudwatch put-metric-data \
  --namespace "DigitalTwin" \
  --metric-name GPUUtilization \
  --value $UTIL \
  --unit Percent

aws cloudwatch put-metric-data \
  --namespace "DigitalTwin" \
  --metric-name GPUMemoryUsed \
  --value $MEM \
  --unit Megabytes
```

## Backup and Recovery

### Snapshot EBS Volume

```bash
aws ec2 create-snapshot \
  --volume-id vol-xxx \
  --description "Digital Twin Backup $(date +%Y-%m-%d)"
```

### AMI Creation

```bash
aws ec2 create-image \
  --instance-id i-xxx \
  --name "digital-twin-ami-$(date +%Y%m%d)" \
  --description "Pre-configured Digital Twin instance"
```

## Cost Optimization

### Spot Instances

For non-critical usage, Spot instances offer 60-70% savings:

```hcl
# In Terraform
resource "aws_spot_instance_request" "gpu" {
  ami           = data.aws_ami.ubuntu_gpu.id
  instance_type = "g5.xlarge"
  spot_price    = "0.50"  # Max price per hour
  # ...
}
```

### Scheduled Start/Stop

```bash
# Create Lambda to stop instance at night
aws events put-rule \
  --name "stop-digital-twin-night" \
  --schedule-expression "cron(0 22 * * ? *)"  # 10 PM UTC
```

### Reserved Instances

For consistent usage (>30% utilization), consider 1-year reserved:
- g5.xlarge: ~$650/month (35% savings)

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| GPU not found | Driver not loaded | Reboot instance |
| Out of memory | Scene too complex | Reduce assets or upgrade instance |
| SSH timeout | Security group | Check allowed IPs |
| Containers crash | API key missing | Check .env file |

### Log Locations

```
/var/log/user-data.log          # Bootstrap log
/var/log/cloud-init-output.log  # Cloud-init
~/digital-twin-robot-NVD/logs/  # Application logs
docker-compose logs             # Container logs
```

## Clean Up

### Terraform

```bash
terraform destroy
```

### CloudFormation

```bash
aws cloudformation delete-stack --stack-name digital-twin-robotics
```

### Manual

1. Terminate EC2 instance
2. Release Elastic IP
3. Delete security group
4. Delete VPC (and associated resources)
5. Delete IAM role/instance profile
