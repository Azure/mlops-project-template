# Bastion

module "bastion" {
  source = "./modules/bastion-host"

  prefix  = var.prefix
  postfix = var.postfix
  env     = var.environment

  rg_name   = module.resource_group.name
  location  = module.resource_group.location
  subnet_id = var.enable_aml_secure_workspace ? azurerm_subnet.snet_bastion[0].id : ""

  enable_aml_secure_workspace = var.enable_aml_secure_workspace

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
  subnet_id         = var.enable_aml_secure_workspace ? azurerm_subnet.snet_default[0].id : ""
  jumphost_username = var.jumphost_username
  jumphost_password = var.jumphost_password

  enable_aml_secure_workspace = var.enable_aml_secure_workspace

  tags = local.tags
}