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

variable "hns_enabled" {
  type        = bool
  description = "Hierarchical namespaces enabled/disabled"
  default     = true
}

variable "firewall_virtual_network_subnet_ids" {
  default = []
}

variable "firewall_bypass" {
  default = ["None"]
}

variable "enable_private_endpoints" {
  type        = bool
  description = "Enable private endpoints for storage account"
  default     = false
}

variable "private_endpoint_subnet_id" {
  type        = string
  description = "Subnet ID for private endpoints"
  default     = ""
}

variable "private_dns_zone_blob_id" {
  type        = string
  description = "Private DNS zone ID for blob storage"
  default     = ""
}

variable "private_dns_zone_file_id" {
  type        = string
  description = "Private DNS zone ID for file storage"
  default     = ""
}

variable "private_dns_zone_dfs_id" {
  type        = string
  description = "Private DNS zone ID for DFS storage"
  default     = ""
}