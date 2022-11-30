locals {
  tags = {
    Owner       = "mlops-v2"
    Project     = "mlops-v2"
    Environment = "${var.environment}"
    Toolkit     = "terraform"
    Name        = "${var.prefix}"
  }
}