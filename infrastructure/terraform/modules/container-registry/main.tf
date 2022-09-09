locals {
  safe_prefix  = replace(var.prefix, "-", "")
  safe_postfix = replace(var.postfix, "-", "")
}

resource "azurerm_container_registry" "cr" {
  name                = "cr${local.safe_prefix}${local.safe_postfix}${var.env}"
  resource_group_name = var.rg_name
  location            = var.location
  sku                 = var.enable_aml_secure_workspace ? "Premium" : "Standard"
  admin_enabled       = true

  tags = var.tags
}

# DNS Zones

resource "azurerm_private_dns_zone" "cr_zone" {
  name                = "privatelink.azurecr.io"
  resource_group_name = var.rg_name

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Linking of DNS zones to Virtual Network

resource "azurerm_private_dns_zone_virtual_network_link" "cr_zone_link" {
  name                  = "${var.prefix}${var.postfix}_link_acr"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.cr_zone[0].name
  virtual_network_id    = var.vnet_id

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Private Endpoint configuration

resource "azurerm_private_endpoint" "cr_pe" {
  name                = "pe-${azurerm_container_registry.cr.name}-acr"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "psc-acr-${var.prefix}-${var.postfix}${var.env}"
    private_connection_resource_id = azurerm_container_registry.cr.id
    subresource_names              = ["registry"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-acr"
    private_dns_zone_ids = [azurerm_private_dns_zone.cr_zone[0].id]
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}