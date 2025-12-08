locals {
  safe_prefix  = replace(var.prefix, "-", "")
  safe_postfix = replace(var.postfix, "-", "")
}

resource "azurerm_container_registry" "cr" {
  name                          = "cr${local.safe_prefix}${local.safe_postfix}${var.env}"
  resource_group_name           = var.rg_name
  location                      = var.location
  sku                           = "Premium"
  admin_enabled                 = false
  public_network_access_enabled = var.enable_private_endpoints ? false : true
  zone_redundancy_enabled       = false
  
  # Network rules configured inline
  # When using private endpoints, default action is Deny and access comes through private endpoint
  # When not using private endpoints, default action is Allow
  network_rule_set {
    default_action = var.enable_private_endpoints ? "Deny" : "Allow"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Private endpoint for Container Registry
resource "azurerm_private_endpoint" "cr_pe" {
  count               = var.enable_private_endpoints ? 1 : 0
  name                = "pe-${azurerm_container_registry.cr.name}"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "psc-${azurerm_container_registry.cr.name}"
    private_connection_resource_id = azurerm_container_registry.cr.id
    subresource_names              = ["registry"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "acr-dns-zone-group"
    private_dns_zone_ids = [var.private_dns_zone_acr_id]
  }

  tags = var.tags
}