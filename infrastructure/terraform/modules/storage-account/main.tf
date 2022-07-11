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

# Private Endpoint configuration

resource "azurerm_private_endpoint" "st_pe_blob" {
  name                = "pe-${azurerm_storage_account.adl_st[0].name}-blob"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "psc-blob-${var.basename}"
    private_connection_resource_id = azurerm_storage_account.adl_st[0].id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-blob"
    private_dns_zone_ids = var.private_dns_zone_ids_blob
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}

resource "azurerm_private_endpoint" "st_pe_file" {
  name                = "pe-${azurerm_storage_account.adl_st[0].name}-file"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "psc-file-${var.basename}"
    private_connection_resource_id = azurerm_storage_account.adl_st[0].id
    subresource_names              = ["file"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-file"
    private_dns_zone_ids = var.private_dns_zone_ids_file
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}