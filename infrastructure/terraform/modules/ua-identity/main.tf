resource "azurerm_user_assigned_identity" "ua_identity" {
  location            = var.location
  name                = "uaid${var.prefix}${var.postfix}${var.env}"
  resource_group_name = var.rg_name

  tags = var.tags
}
