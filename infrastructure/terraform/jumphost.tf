# Bastion

module "bastion" {
  source = "./modules/bastion-host"

  prefix  = var.prefix
  postfix = var.postfix
  env     = var.environment

  rg_name   = module.resource_group.name
  location  = module.resource_group.location
  subnet_id = azurerm_subnet.snet_bastion.id

  tags = local.tags
}

# Virtual machine

module "virtual_machine_jumphost" {
  source = "./modules/virtual-machine"

  prefix  = var.prefix
  postfix = var.postfix
  env     = var.environment

  rg_name           = module.resource_group.name
  location          = module.resource_group.location
  subnet_id         = azurerm_subnet.snet_default.id
  jumphost_username = var.jumphost_username
  jumphost_password = var.jumphost_password

  tags = local.tags
}