resource "azurerm_application_insights" "appi" {
  name                = "appi-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  application_type    = "web"
  workspace_id        = var.log_analytics_workspace_id 

  tags = var.tags
}
