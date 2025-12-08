data "azurerm_client_config" "current" {}

data "http" "ip" {
  url = "https://ifconfig.me"
}

locals {
  safe_prefix  = replace(var.prefix, "-", "")
  safe_postfix = replace(var.postfix, "-", "")
}

resource "azurerm_storage_account" "st" {
  name                          = "st${local.safe_prefix}${local.safe_postfix}${var.env}"
  resource_group_name           = var.rg_name
  location                      = var.location
  account_tier                  = "Standard"
  account_replication_type      = "LRS"
  account_kind                  = "StorageV2"
  is_hns_enabled                = var.hns_enabled
  min_tls_version               = "TLS1_2"
  public_network_access_enabled = var.enable_private_endpoints ? false : true
  shared_access_key_enabled     = false  # Disable key-based authentication, use managed identity instead
  
  blob_properties {
    delete_retention_policy {
      days = 7
    }
    container_delete_retention_policy {
      days = 7
    }
  }

  tags = var.tags
  
}

# Virtual Network & Firewall configuration

resource "azurerm_storage_account_network_rules" "firewall_rules" {
  storage_account_id = azurerm_storage_account.st.id

  default_action             = var.enable_private_endpoints ? "Deny" : "Allow"
  ip_rules                   = [] # [data.http.ip.body]
  virtual_network_subnet_ids = var.firewall_virtual_network_subnet_ids
  bypass                     = var.firewall_bypass
}

# Private endpoints for storage account
resource "azurerm_private_endpoint" "st_blob_pe" {
  count               = var.enable_private_endpoints ? 1 : 0
  name                = "pe-${azurerm_storage_account.st.name}-blob"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "psc-${azurerm_storage_account.st.name}-blob"
    private_connection_resource_id = azurerm_storage_account.st.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "blob-dns-zone-group"
    private_dns_zone_ids = [var.private_dns_zone_blob_id]
  }

  tags = var.tags
}

resource "azurerm_private_endpoint" "st_file_pe" {
  count               = var.enable_private_endpoints ? 1 : 0
  name                = "pe-${azurerm_storage_account.st.name}-file"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "psc-${azurerm_storage_account.st.name}-file"
    private_connection_resource_id = azurerm_storage_account.st.id
    subresource_names              = ["file"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "file-dns-zone-group"
    private_dns_zone_ids = [var.private_dns_zone_file_id]
  }

  tags = var.tags
}

# Private endpoint for DFS (Data Lake Gen2) - only if HNS is enabled
resource "azurerm_private_endpoint" "st_dfs_pe" {
  count               = var.enable_private_endpoints && var.hns_enabled ? 1 : 0
  name                = "pe-${azurerm_storage_account.st.name}-dfs"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "psc-${azurerm_storage_account.st.name}-dfs"
    private_connection_resource_id = azurerm_storage_account.st.id
    subresource_names              = ["dfs"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "dfs-dns-zone-group"
    private_dns_zone_ids = [var.private_dns_zone_dfs_id]
  }

  tags = var.tags
}
