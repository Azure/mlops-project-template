resource "azurerm_synapse_workspace" "feathr_synapse_workspace" {
  name                                 = "sy${var.prefix}-${var.postfix}${var.env}"
  resource_group_name                  = var.rg_name
  location                             = var.location
  storage_data_lake_gen2_filesystem_id = var.storage_account_id
  sql_administrator_login              = var.sql_admin_user
  sql_administrator_login_password     = var.sql_admin_password

  tags = var.tags
}


resource "azurerm_synapse_spark_pool" "feathr_synapse_sparkpool" {
  name                 = "sp${var.env}"
  synapse_workspace_id = azurerm_synapse_workspace.feathr_synapse_workspace.id
  node_size_family     = "MemoryOptimized"
  node_size            = "Small"
  cache_size           = 100

  auto_scale {
    max_node_count = 5
    min_node_count = 3
  }

  auto_pause {
    delay_in_minutes = 15
  }

  tags = var.tags
}
