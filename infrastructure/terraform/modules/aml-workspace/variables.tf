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

variable "storage_account_id" {
  type        = string
  description = "The ID of the Storage Account linked to AML workspace"
}

variable "key_vault_id" {
  type        = string
  description = "The ID of the Key Vault linked to AML workspace"
}

variable "application_insights_id" {
  type        = string
  description = "The ID of the Application Insights linked to AML workspace"
}

variable "container_registry_id" {
  type        = string
  description = "The ID of the Container Registry linked to AML workspace"
}

variable "enable_aml_computecluster" {
  description = "Variable to enable or disable AML compute cluster"
  default     = false
}

variable "storage_account_name" {
  type        = string
  description = "The Name of the Storage Account linked to AML workspace"
}

variable "enable_private_endpoints" {
  type        = bool
  description = "Enable private endpoints for ML Workspace"
  default     = false
}

variable "private_endpoint_subnet_id" {
  type        = string
  description = "Subnet ID for private endpoints"
  default     = ""
}

variable "private_dns_zone_aml_api_id" {
  type        = string
  description = "Private DNS zone ID for Azure ML API"
  default     = ""
}

variable "private_dns_zone_aml_notebooks_id" {
  type        = string
  description = "Private DNS zone ID for Azure ML Notebooks"
  default     = ""
}

variable "github_actions_service_principal_id" {
  type        = string
  description = "The object ID of the GitHub Actions service principal for role assignments"
  default     = ""
}