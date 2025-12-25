# Digital Twin Robotics Lab - Terraform Configuration
# Provider: AWS with GPU-enabled EC2 instance

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu_gpu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.environment_name}-vpc"
    Environment = var.environment_name
    Project     = "Digital Twin Robotics Lab"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${var.environment_name}-igw"
    Environment = var.environment_name
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.environment_name}-public-subnet"
    Environment = var.environment_name
  }
}

# Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${var.environment_name}-public-rt"
    Environment = var.environment_name
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Security Group
resource "aws_security_group" "instance" {
  name        = "${var.environment_name}-sg"
  description = "Security group for Digital Twin Robotics Lab"
  vpc_id      = aws_vpc.main.id

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Foxglove Studio Web
  ingress {
    description = "Foxglove Studio"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Foxglove WebSocket Bridge
  ingress {
    description = "Foxglove Bridge"
    from_port   = 8765
    to_port     = 8765
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Isaac Sim Livestream
  ingress {
    description = "Isaac Sim Livestream"
    from_port   = 8211
    to_port     = 8211
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Isaac Sim WebRTC
  ingress {
    description = "Isaac Sim WebRTC"
    from_port   = 49100
    to_port     = 49100
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # ROS 2 DDS Discovery
  ingress {
    description = "ROS 2 DDS"
    from_port   = 7400
    to_port     = 7500
    protocol    = "udp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Allow all outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.environment_name}-sg"
    Environment = var.environment_name
  }
}

# IAM Role
resource "aws_iam_role" "instance" {
  name = "${var.environment_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.environment_name}-ec2-role"
    Environment = var.environment_name
  }
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.environment_name}-instance-profile"
  role = aws_iam_role.instance.name
}

# EC2 Instance
resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.ubuntu_gpu.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  root_block_device {
    volume_size           = var.volume_size
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
    delete_on_termination = true

    tags = {
      Name        = "${var.environment_name}-root-volume"
      Environment = var.environment_name
    }
  }

  user_data = base64encode(templatefile("${path.module}/bootstrap.sh", {
    ngc_api_key = var.ngc_api_key
    nim_api_key = var.nim_api_key
  }))

  tags = {
    Name        = "${var.environment_name}-gpu-instance"
    Environment = var.environment_name
    Project     = "Digital Twin Robotics Lab"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}

# Elastic IP
resource "aws_eip" "instance" {
  domain   = "vpc"
  instance = aws_instance.gpu.id

  tags = {
    Name        = "${var.environment_name}-eip"
    Environment = var.environment_name
  }

  depends_on = [aws_internet_gateway.main]
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "instance" {
  name              = "/aws/ec2/${var.environment_name}"
  retention_in_days = 7

  tags = {
    Name        = "${var.environment_name}-logs"
    Environment = var.environment_name
  }
}
