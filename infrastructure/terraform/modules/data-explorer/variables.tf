variable "rg_name" {
  type        = string
  description = "Resource group name"
}

variable "location" {
  type        = string
  description = "Location of the resource group"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A mapping of tags which should be assigned to the deployed resource"
}

variable "prefix" {
  type        = string
  description = "Prefix for the module name"
}

variable "postfix" {
  type        = string
  description = "Postfix for the module name"
}

variable "env" {
  type        = string
  description = "Environment prefix"
}

variable "key_vault_id" {
  type        = string
  description = "The ID of the Key Vault linked to AML workspace"
}

variable "enable_monitoring" {
  description = "Variable to enable or disable AML compute cluster"
  default     = false
}

variable "client_secret" {
  description = "client secret"
  default     = false
}