# AWS Region
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

# Environment name
variable "environment_name" {
  description = "Environment name prefix for all resources"
  type        = string
  default     = "digital-twin-robotics"
}

# EC2 Instance Type
variable "instance_type" {
  description = "EC2 instance type with GPU"
  type        = string
  default     = "g5.xlarge"

  validation {
    condition = contains([
      "g4dn.xlarge",  # T4 16GB - Budget option
      "g4dn.2xlarge", # T4 16GB + more CPU/RAM
      "g5.xlarge",    # A10G 24GB - Recommended
      "g5.2xlarge",   # A10G 24GB + more CPU/RAM
      "p3.2xlarge",   # V100 16GB - High performance
    ], var.instance_type)
    error_message = "Instance type must be a GPU instance: g4dn.xlarge, g4dn.2xlarge, g5.xlarge, g5.2xlarge, or p3.2xlarge"
  }
}

# Key Pair
variable "key_pair_name" {
  description = "Name of existing EC2 KeyPair for SSH access"
  type        = string
}

# Volume Size
variable "volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 200

  validation {
    condition     = var.volume_size >= 100 && var.volume_size <= 1000
    error_message = "Volume size must be between 100 and 1000 GB"
  }
}

# SSH Access
variable "allowed_ssh_cidr" {
  description = "CIDR block allowed for SSH access (your IP/32 recommended)"
  type        = string
  default     = "0.0.0.0/0"
}

# NVIDIA API Keys
variable "ngc_api_key" {
  description = "NVIDIA NGC API Key for container access"
  type        = string
  default     = ""
  sensitive   = true
}

variable "nim_api_key" {
  description = "NVIDIA NIM API Key for LLM access"
  type        = string
  default     = ""
  sensitive   = true
}
