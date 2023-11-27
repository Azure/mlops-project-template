resource "azurerm_log_analytics_workspace" "log" {
  name                = "log-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  sku                 = "PerGB2018"

  tags = var.tags
}
