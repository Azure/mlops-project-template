resource "azurerm_bastion_host" "bas" {
  name                = "bas-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name

  sku                = "Standard"
  copy_paste_enabled = false
  file_copy_enabled  = false

  ip_configuration {
    name                 = "configuration"
    subnet_id            = var.subnet_id
    public_ip_address_id = azurerm_public_ip.pip.id
  }

  tags = var.tags
}

resource "azurerm_public_ip" "pip" {
  name                = "pip-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = var.tags
}