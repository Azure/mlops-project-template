variable "rg_name" {
  type        = string
  description = "Resource group name"
}

variable "location" {
  type        = string
  default     = "North Europe"
  description = "Location of the Resource Group"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A mapping of tags which should be assigned to the Resource Group"
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

variable "sql_admin_user" {
  type        = string
  description = "SQL admin user name"
}

variable "sql_admin_password" {
  type        = string
  description = "SQL admin password"
}
