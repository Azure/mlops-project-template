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

variable "feathr_app_image" {
  type        = string
  description = "Feathr app image name to pull from docker registry"
}

variable "feathr_app_image_tag" {
  type        = string
  description = "Feathr app image tag to pull from docker registry"
}

variable "react_enable_rbac" {
  type        = bool
  description = "Enable RBAC for the Feathr React App"
}

variable "tenant_id" {
  type        = string
  description = "Tenant ID"
}

variable "aad_client_id" {
  type        = string
  description = "AAD Client ID"
}

variable "uaid_client_id" {
  type        = string
  description = "User assigned identity client ID for the Feature store"
}

variable "sql_admin_user" {
  type        = string
  description = "SQL admin user name"
}

variable "sql_admin_password" {
  type        = string
  description = "SQL admin password"
}

variable "mssql_server_name" {
  type        = string
  description = "SQL server name"
}

variable "mssql_db_name" {
  type        = string
  description = "SQL database name"
}

variable "appi_instrumentation_key" {
  type        = string
  description = "Application Insights instrumentation key"
}
  
variable "appi_connection_string" {
  type        = string
  description = "Application Insights connection string"
}
