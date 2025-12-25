# Instance Information
output "instance_id" {
  description = "EC2 Instance ID"
  value       = aws_instance.gpu.id
}

output "public_ip" {
  description = "Public IP address"
  value       = aws_eip.instance.public_ip
}

output "public_dns" {
  description = "Public DNS name"
  value       = aws_eip.instance.public_dns
}

# Connection Information
output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.key_pair_name}.pem ubuntu@${aws_eip.instance.public_ip}"
}

output "ssh_config" {
  description = "SSH config entry"
  value       = <<-EOT
    Host digital-twin
      HostName ${aws_eip.instance.public_ip}
      User ubuntu
      IdentityFile ~/.ssh/${var.key_pair_name}.pem
  EOT
}

# Service URLs
output "foxglove_studio_url" {
  description = "Foxglove Studio Web URL"
  value       = "http://${aws_eip.instance.public_ip}:8080"
}

output "foxglove_bridge_url" {
  description = "Foxglove WebSocket Bridge URL"
  value       = "ws://${aws_eip.instance.public_ip}:8765"
}

output "isaac_sim_url" {
  description = "Isaac Sim Livestream URL"
  value       = "http://${aws_eip.instance.public_ip}:8211"
}

# Cost Estimate
output "estimated_cost" {
  description = "Estimated hourly cost (USD)"
  value       = {
    "g4dn.xlarge"  = "$0.526/hour (~$380/month if 24/7)"
    "g4dn.2xlarge" = "$0.752/hour (~$540/month if 24/7)"
    "g5.xlarge"    = "$1.006/hour (~$725/month if 24/7)"
    "g5.2xlarge"   = "$1.212/hour (~$875/month if 24/7)"
    "p3.2xlarge"   = "$3.060/hour (~$2200/month if 24/7)"
    "selected"     = var.instance_type
  }
}

# Quick Start Instructions
output "quick_start" {
  description = "Quick start instructions"
  value       = <<-EOT

    ╔══════════════════════════════════════════════════════════════════╗
    ║         Digital Twin Robotics Lab - Deployment Complete          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. Wait for setup to complete (2-5 minutes):                    ║
    ║     ${output.ssh_command.value}
    ║     tail -f /var/log/user-data.log                               ║
    ║                                                                  ║
    ║  2. Add your NVIDIA API keys:                                    ║
    ║     cd digital-twin-robot-NVD                                    ║
    ║     nano .env                                                    ║
    ║                                                                  ║
    ║  3. Start the platform:                                          ║
    ║     docker-compose up -d                                         ║
    ║                                                                  ║
    ║  4. Access services:                                             ║
    ║     - Foxglove: http://${aws_eip.instance.public_ip}:8080        ║
    ║     - Isaac Sim: http://${aws_eip.instance.public_ip}:8211       ║
    ║                                                                  ║
    ║  Remember to STOP the instance when not in use to save costs!    ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
  EOT
}
