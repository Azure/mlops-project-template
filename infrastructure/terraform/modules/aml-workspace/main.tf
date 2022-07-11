resource "azurerm_machine_learning_workspace" "mlw" {
  name                    = "mlw-${var.prefix}-${var.postfix}${var.env}"
  location                = var.location
  resource_group_name     = var.rg_name
  application_insights_id = var.application_insights_id
  key_vault_id            = var.key_vault_id
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id

  sku_name = "Basic"

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Compute cluster

resource "azurerm_machine_learning_compute_cluster" "mlw_compute_cluster" {
  name                          = "cpu-cluster"
  location                      = var.location
  vm_priority                   = "LowPriority"
  vm_size                       = "Standard_DS3_v2"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.mlw.id
  subnet_resource_id            = var.enable_aml_secure_workspace ? var.subnet_training_id : ""

  count = var.enable_aml_computecluster ? 1 : 0

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 4
    scale_down_nodes_after_idle_duration = "PT120S" # 120 seconds
  }
}

# DNS Zones

resource "azurerm_private_dns_zone" "mlw_zone_api" {
  name                = "privatelink.api.azureml.ms"
  resource_group_name = var.rg_name

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_private_dns_zone" "mlw_zone_notebooks" {
  name                = "privatelink.notebooks.azure.net"
  resource_group_name = var.rg_name

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Linking of DNS zones to Virtual Network

resource "azurerm_private_dns_zone_virtual_network_link" "mlw_zone_api_link" {
  name                  = "${var.prefix}${var.postfix}_link_api"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.mlw_zone_api[0].name
  virtual_network_id    = var.vnet_id

  count = var.enable_aml_secure_workspace ? 1 : 0
}

resource "azurerm_private_dns_zone_virtual_network_link" "mlw_zone_notebooks_link" {
  name                  = "${var.prefix}${var.postfix}_link_notebooks"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.mlw_zone_notebooks[0].name
  virtual_network_id    = var.vnet_id

  count = var.enable_aml_secure_workspace ? 1 : 0
}

# Private Endpoint configuration

resource "azurerm_private_endpoint" "mlw_pe" {
  name                = "pe-${azurerm_machine_learning_workspace.mlw.name}-amlw"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.subnet_default_id

  private_service_connection {
    name                           = "psc-aml-${var.prefix}-${var.postfix}${var.env}"
    private_connection_resource_id = azurerm_machine_learning_workspace.mlw.id
    subresource_names              = ["amlworkspace"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "private-dns-zone-group-ws"
    private_dns_zone_ids = [azurerm_private_dns_zone.mlw_zone_api[0].id, azurerm_private_dns_zone.mlw_zone_notebooks[0].id]
  }

  count = var.enable_aml_secure_workspace ? 1 : 0

  tags = var.tags
}
