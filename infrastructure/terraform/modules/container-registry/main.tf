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

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Network rule set for Container Registry
resource "azurerm_container_registry_network_rule_set" "cr_acl" {
  container_registry_id = azurerm_container_registry.cr.id
  default_action        = var.enable_private_endpoints ? "Deny" : "Allow"

  dynamic "virtual_network" {
    for_each = var.firewall_virtual_network_subnet_ids
    content {
      subnet_id = virtual_network.value
    }
  }
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