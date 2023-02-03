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

variable "enable_aml_secure_workspace" {
  description = "Variable to enable or disable AML secure workspace"
}

variable "vnet_id" {
  type        = string
  description = "The ID of the vnet that should be linked to the DNS zone"
}

variable "subnet_id" {
  type        = string
  description = "The ID of the subnet from which private IP addresses will be allocated for this Private Endpoint"
}

variable "fs_onlinestore_conn_name" {
  type        = string
  default     = ""
  description = "feature store: Name of the secret for the online store connection string"
}


variable "fs_onlinestore_conn" {
  type        = string
  default     = ""
  description = "feature store: Online store connection string"
}

variable "enable_feature_store" {
  type        = bool
  default     = false
  description = "Enable feature store deployment (additional secrets deployed) to key vault"
}

variable "uaid_principal_id" {
  type        = string
  default     = ""
  description = "Principal ID of the User Assigned Identity for feature store"
}

variable "uaid_tenant_id" {
  type        = string
  default     = ""
  description = "Tenant ID of the User Assigned Identity for feature store"
}
