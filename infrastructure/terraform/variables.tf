variable "location" {
  type        = string
  description = "Location of the resource group and modules"
}

variable "prefix" {
  type        = string
  description = "Prefix for module names"
}

variable "environment" {
  type        = string
  description = "Environment information"
}

variable "postfix" {
  type        = string
  description = "Postfix for module names"
}

variable "enable_aml_computecluster" {
  description = "Variable to enable or disable AML compute cluster"
  default = true
}

variable "enable_aml_secure_workspace" {
  description = "Variable to enable or disable AML secure workspace"
  default = false
}

variable "jumphost_username" {
  type        = string
  description = "VM username"
  default     = "azureuser"
}

variable "jumphost_password" {
  type        = string
  description = "VM password"
  default     = "ThisIsNotVerySecure!"
}

variable "enable_monitoring" {
  description = "Variable to enable or disable Monitoring"
}

variable "client_secret" {
  description = "Service Principal Secret"
}
