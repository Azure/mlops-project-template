# Virtual network

resource "azurerm_virtual_network" "adl_vnet" {
  source = "./modules/virtual-network"

  basename      = local.basename
  rg_name       = module.resource_group.name
  location      = module.resource_group.location
  address_space = ["10.0.0.0/16"]

  tags = local.tags
}

# Subnets

module "subnet_default" {
  source = "./modules/subnet"

  name                                           = "snet-${var.prefix}-${var.postfix}-default"
  rg_name                                        = module.resource_group.name
  vnet_name                                      = module.virtual_network.name
  address_prefixes                               = ["10.0.1.0/24"]
  enforce_private_link_endpoint_network_policies = true
}

module "subnet_bastion" {
  source = "./modules/subnet"

  name             = "AzureBastionSubnet"
  rg_name          = module.resource_group.name
  vnet_name        = module.virtual_network.name
  address_prefixes = ["10.0.10.0/27"]
}

module "subnet_compute" {
  source = "./modules/subnet"

  name                                           = "snet-${var.prefix}-${var.postfix}-compute"
  rg_name                                        = module.resource_group.name
  vnet_name                                      = module.virtual_network.name
  address_prefixes                               = ["10.0.2.0/24"]
  enforce_private_link_endpoint_network_policies = true
}

# Network security groups

module "network_security_group_training" {
  source = "./modules/network-security-group"

  basename = local.basename
  rg_name  = module.resource_group.name
  location = var.location

  tags = local.tags
}

# Network security rules

module "network_security_rule_training_batchnodemanagement" {
  source = "./modules/network-security-rule"

  name                       = "BatchNodeManagement"
  priority                   = 100
  direction                  = "Inbound"
  access                     = "Allow"
  protocol                   = "Tcp"
  source_port_range          = "*"
  destination_port_range     = "29876-29877"
  source_address_prefix      = "BatchNodeManagement"
  destination_address_prefix = "*"

  rg_name                     = module.resource_group.name
  network_security_group_name = module.network_security_group_training.name
}

module "network_security_rule_training_azuremachinelearning" {
  source = "./modules/network-security-rule"

  name                       = "AzureMachineLearning"
  priority                   = 110
  direction                  = "Inbound"
  access                     = "Allow"
  protocol                   = "Tcp"
  source_port_range          = "*"
  destination_port_range     = "44224"
  source_address_prefix      = "AzureMachineLearning"
  destination_address_prefix = "*"

  rg_name                     = module.resource_group.name
  network_security_group_name = module.network_security_group_training.name
}

# NSG associations

module "subnet_network_security_group_association_training" {
  source = "./modules/subnet-network-security-group-association"

  subnet_id                 = module.subnet_compute.id
  network_security_group_id = module.network_security_group_training.id
}

# User Defined Routes

module "route_table_training" {
  source = "./modules/route-table"

  basename = local.basename
  location = module.resource_group.location
  rg_name  = module.resource_group.name
}

module "route_training_internet" {
  source = "./modules/route"

  name             = "Internet"
  rg_name          = module.resource_group.name
  route_table_name = module.route_table_training.name
  address_prefix   = "0.0.0.0/0"
  next_hop_type    = "Internet"
}

module "route_training_azureml" {
  source = "github.com/microsoft/azure-labs-modules/terraform/route"

  name             = "AzureMLRoute"
  rg_name          = module.resource_group.name
  route_table_name = module.route_table_training.name
  address_prefix   = "AzureMachineLearning"
  next_hop_type    = "Internet"
}

module "route_training_batch" {
  source = "github.com/microsoft/azure-labs-modules/terraform/route"

  name             = "BatchRoute"
  rg_name          = module.resource_group.name
  route_table_name = module.route_table_training.name
  address_prefix   = "BatchNodeManagement"
  next_hop_type    = "Internet"
}

# UDR associations

module "subnet_route_table_association_training" {
  source = "github.com/microsoft/azure-labs-modules/terraform/subnet-route-table-association"

  subnet_id      = module.subnet_compute.id
  route_table_id = module.route_table_training.id
}