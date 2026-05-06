variable "rg_name" {
  type        = string
  description = "Resource group name"
}

variable "location" {
  type        = string
  description = "Azure region for resources"
}

variable "prefix" {
  type        = string
  description = "Prefix for resource names"
}

variable "postfix" {
  type        = string
  description = "Postfix for resource names"
}

variable "env" {
  type        = string
  description = "Environment name"
}

variable "vnet_address_space" {
  type        = string
  description = "Address space for the virtual network"
  default     = "10.0.0.0/16"
}

variable "training_subnet_address_prefix" {
  type        = string
  description = "Address prefix for training subnet"
  default     = "10.0.0.0/24"
}

variable "endpoints_subnet_address_prefix" {
  type        = string
  description = "Address prefix for private endpoints subnet"
  default     = "10.0.1.0/24"
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply to resources"
  default     = {}
}
