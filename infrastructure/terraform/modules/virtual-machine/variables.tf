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

variable "jumphost_username" {
  type        = string
  description = "VM username"
}

variable "jumphost_password" {
  type        = string
  description = "VM password"
}

variable "subnet_id" {
  type        = string
  description = "Subnet ID for the virtual machine"
}

variable "enable_aml_secure_workspace" {
  description = "Variable to enable or disable AML secure workspace"
}