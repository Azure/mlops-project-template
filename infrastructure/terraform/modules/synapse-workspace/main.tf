data "azurerm_client_config" "current" {}

data "http" "ip" {
  url = "http://ipv4.icanhazip.com"
}

resource "azurerm_synapse_workspace" "feathr_synapse_workspace" {
  name                                 = "sy${var.prefix}-${var.postfix}${var.env}"
  resource_group_name                  = var.rg_name
  location                             = var.location
  storage_data_lake_gen2_filesystem_id = var.storage_account_id
  sql_administrator_login              = var.sql_admin_user
  sql_administrator_login_password     = var.sql_admin_password

  tags = var.tags
}

# Allow deployment VM to access the Synapse workspace to make changes
resource "azurerm_synapse_firewall_rule" "allow_deployment_vm" {
  name                 = "AllowDeploymetVM"
  synapse_workspace_id = azurerm_synapse_workspace.feathr_synapse_workspace.id
  start_ip_address     = "${chomp(data.http.ip.body)}"
  end_ip_address       = "${chomp(data.http.ip.body)}"
}

resource "azurerm_synapse_spark_pool" "feathr_synapse_sparkpool" {
  name                 = "sp${var.env}"
  synapse_workspace_id = azurerm_synapse_workspace.feathr_synapse_workspace.id
  node_size_family     = "MemoryOptimized"
  node_size            = "Small"
  cache_size           = 100
  spark_version        = var.spark_version

  auto_scale {
    max_node_count = 5
    min_node_count = 3
  }

  auto_pause {
    delay_in_minutes = 15
  }

  tags = var.tags
}

resource "azurerm_synapse_workspace_aad_admin" "synapse_aad_admin" {
  synapse_workspace_id = azurerm_synapse_workspace.feathr_synapse_workspace.id
  login                = "AzureAD Admin"
  object_id            = var.priviledged_object_id
  tenant_id            = data.azurerm_client_config.current.tenant_id
}

# introducing as firewall rule needs time to take effect: https://github.com/hashicorp/terraform-provider-azurerm/issues/13510
resource "time_sleep" "wait_60_seconds" {
  depends_on = [azurerm_synapse_firewall_rule.allow_deployment_vm]

  create_duration = "60s"
}

resource "azurerm_synapse_role_assignment" "synapse_workspace_admin" {
  synapse_workspace_id = azurerm_synapse_workspace.feathr_synapse_workspace.id
  role_name            = "Synapse Administrator"
  principal_id         = var.priviledged_object_id

  depends_on = [azurerm_synapse_firewall_rule.allow_deployment_vm, time_sleep.wait_60_seconds]
}
