# Virtual network

resource "azurerm_virtual_network" "vnet_default" {
  name                = "vnet-${var.prefix}-${var.postfix}${var.environment}"
  resource_group_name = module.resource_group.name
  location            = module.resource_group.location
  address_space       = ["10.0.0.0/16"]

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = local.tags
}

# Subnets

resource "azurerm_subnet" "snet_default" {
  name                                           = "snet-${var.prefix}-${var.postfix}${var.environment}-default"
  resource_group_name                            = module.resource_group.name
  virtual_network_name                           = azurerm_virtual_network.vnet_default[0].name
  address_prefixes                               = ["10.0.1.0/24"]
  enforce_private_link_endpoint_network_policies = true

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_subnet" "snet_bastion" {
  name                 = "AzureBastionSubnet"
  resource_group_name  = module.resource_group.name
  virtual_network_name = azurerm_virtual_network.vnet_default[0].name
  address_prefixes     = ["10.0.10.0/27"]

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_subnet" "snet_training" {
  name                                           = "snet-${var.prefix}-${var.postfix}${var.environment}-training"
  resource_group_name                            = module.resource_group.name
  virtual_network_name                           = azurerm_virtual_network.vnet_default[0].name
  address_prefixes                               = ["10.0.2.0/24"]
  enforce_private_link_endpoint_network_policies = true

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Network security groups

resource "azurerm_network_security_group" "nsg_training" {
  name                = "nsg-${var.prefix}-${var.postfix}${var.environment}-training"
  location            = module.resource_group.location
  resource_group_name = module.resource_group.name

  security_rule {
    name                       = "BatchNodeManagement"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "29876-29877"
    source_address_prefix      = "BatchNodeManagement"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AzureMachineLearning"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "44224"
    source_address_prefix      = "AzureMachineLearning"
    destination_address_prefix = "*"
  }

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_subnet_network_security_group_association" "nsg-training-link" {
  subnet_id                 = azurerm_subnet.snet_training[0].id
  network_security_group_id = azurerm_network_security_group.nsg_training[0].id

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# User Defined Routes

resource "azurerm_route_table" "rt_training" {
  name                = "rt-${var.prefix}-${var.postfix}${var.environment}-training"
  location            = module.resource_group.location
  resource_group_name = module.resource_group.name

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_route" "route_training_internet" {
  name                = "Internet"
  resource_group_name = module.resource_group.name
  route_table_name    = azurerm_route_table.rt_training[0].name
  address_prefix      = "0.0.0.0/0"
  next_hop_type       = "Internet"

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_route" "route_training_aml" {
  name                = "AzureMLRoute"
  resource_group_name = module.resource_group.name
  route_table_name    = azurerm_route_table.rt_training[0].name
  address_prefix      = "AzureMachineLearning"
  next_hop_type       = "Internet"

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_route" "route_training_batch" {
  name                = "BatchRoute"
  resource_group_name = module.resource_group.name
  route_table_name    = azurerm_route_table.rt_training[0].name
  address_prefix      = "BatchNodeManagement"
  next_hop_type       = "Internet"

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_subnet_route_table_association" "rt_training_link" {
  subnet_id      = azurerm_subnet.snet_training[0].id
  route_table_id = azurerm_route_table.rt_training[0].id

  count = var.enable_aml_secure_workspace ? 1 : 0
}