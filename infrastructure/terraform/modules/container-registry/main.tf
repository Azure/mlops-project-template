locals {
  safe_prefix  = replace(var.prefix, "-", "")
  safe_postfix = replace(var.postfix, "-", "")
}

resource "azurerm_container_registry" "cr" {
  name                = "cr${local.safe_prefix}${local.safe_postfix}${var.env}"
  resource_group_name = var.rg_name
  location            = var.location
  sku                 = "Standard"
  admin_enabled       = true

  tags = var.tags
}


# Private Endpoint configuration

resource "azurerm_private_endpoint" "cr_pe" {
  name                = "pe-${azurerm_container_registry.adl_cr.name}-acr"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "psc-acr-${var.basename}"
    private_connection_resource_id = azurerm_container_registry.adl_cr.id
    subresource_names              = ["registry"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-acr"
    private_dns_zone_ids = var.private_dns_zone_ids
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}