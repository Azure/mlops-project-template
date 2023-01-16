resource "azurerm_redis_cache" "redis_cache" {
  name                = "rd${var.prefix}${var.postfix}${var.env}"
  resource_group_name = var.rg_name
  location            = var.location
  capacity            = 2
  family              = "C"
  sku_name            = "Standard"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  tags = var.tags
}
