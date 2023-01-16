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

variable "priviledged_object_id" {
  type        = string
  default     = ""
  description = "Object ID of the user or service principal that will ge granted priviledges in keyvault for feature store"
}

