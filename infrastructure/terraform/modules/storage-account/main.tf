data "azurerm_client_config" "current" {}

data "http" "ip" {
  url = "https://ifconfig.me"
}

locals {
  safe_prefix  = replace(var.prefix, "-", "")
  safe_postfix = replace(var.postfix, "-", "")
}

resource "azurerm_storage_account" "st" {
  name                     = "st${local.safe_prefix}${local.safe_postfix}${var.env}"
  resource_group_name      = var.rg_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = var.hns_enabled

  tags = var.tags
}

# Virtual Network & Firewall configuration

resource "azurerm_storage_account_network_rules" "firewall_rules" {
  resource_group_name  = var.rg_name
  storage_account_name = azurerm_storage_account.st.name

  default_action             = "Allow"
  ip_rules                   = [] # [data.http.ip.body]
  virtual_network_subnet_ids = var.firewall_virtual_network_subnet_ids
  bypass                     = var.firewall_bypass
}

# DNS Zones

resource "azurerm_private_dns_zone" "st_zone_blob" {
  name                = "privatelink.blob.core.windows.net"
  resource_group_name = var.rg_name

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_private_dns_zone" "st_zone_file" {
  name                = "privatelink.file.core.windows.net"
  resource_group_name = var.rg_name

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Linking of DNS zones to Virtual Network

resource "azurerm_private_dns_zone_virtual_network_link" "st_zone_link_blob" {
  name                  = "${var.prefix}${var.postfix}_link_st_blob"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.st_zone_blob[0].name
  virtual_network_id    = var.vnet_id

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_private_dns_zone_virtual_network_link" "st_zone_link_file" {
  name                  = "${var.prefix}${var.postfix}_link_st_file"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.st_zone_file[0].name
  virtual_network_id    = var.vnet_id

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Private Endpoint configuration

resource "azurerm_private_endpoint" "st_pe_blob" {
  name                = "pe-${azurerm_storage_account.st.name}-blob"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "psc-blob-${var.prefix}-${var.postfix}${var.env}"
    private_connection_resource_id = azurerm_storage_account.st.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-blob"
    private_dns_zone_ids = [azurerm_private_dns_zone.st_zone_blob[0].id]
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}

resource "azurerm_private_endpoint" "st_pe_file" {
  name                = "pe-${azurerm_storage_account.st.name}-file"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "psc-file-${var.prefix}-${var.postfix}${var.env}"
    private_connection_resource_id = azurerm_storage_account.st.id
    subresource_names              = ["file"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-file"
    private_dns_zone_ids = [azurerm_private_dns_zone.st_zone_file[0].id]
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}

# Data Lake Gen2 file system required for synapse workspace
resource "azurerm_storage_data_lake_gen2_filesystem" "st_filesystem" {
  name               = "dl${local.safe_prefix}${local.safe_postfix}${var.env}"
  storage_account_id = azurerm_storage_account.st.id

  count = var.enable_feature_store ? 1 : 0
}
