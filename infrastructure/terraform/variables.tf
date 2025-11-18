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
}

variable "enable_monitoring" {
  description = "Variable to enable or disable Monitoring"
  type        = bool
  default     = false
}

variable "github_actions_service_principal_id" {
  type        = string
  description = "The object ID of the GitHub Actions service principal for role assignments"
  default     = ""
}

variable "enable_private_endpoints" {
  type        = bool
  description = "Enable private endpoints and VNet isolation for Azure ML workspace and dependent services"
  default     = false
}

variable "vnet_address_space" {
  type        = string
  description = "Address space for the virtual network (only used if enable_private_endpoints is true)"
  default     = "10.0.0.0/16"
}

variable "training_subnet_address_prefix" {
  type        = string
  description = "Address prefix for training/compute subnet (only used if enable_private_endpoints is true)"
  default     = "10.0.0.0/24"
}

variable "endpoints_subnet_address_prefix" {
  type        = string
  description = "Address prefix for private endpoints subnet (only used if enable_private_endpoints is true)"
  default     = "10.0.1.0/24"
}
