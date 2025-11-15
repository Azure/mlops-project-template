resource "azurerm_user_assigned_identity" "mlw_uai" {
  name                = "uai-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  tags                = var.tags
}

# Grant the user-assigned managed identity access to storage account
resource "azurerm_role_assignment" "mlw_uai_storage_blob_data_reader" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

resource "azurerm_role_assignment" "mlw_uai_storage_blob_data_contributor" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

resource "azurerm_role_assignment" "mlw_uai_storage_account_contributor" {
  scope                = var.storage_account_id
  role_definition_name = "Contributor"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

# Grant the user-assigned managed identity access to Key Vault
resource "azurerm_role_assignment" "mlw_uai_keyvault_reader" {
  scope                = var.key_vault_id
  role_definition_name = "Reader"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

resource "azurerm_role_assignment" "mlw_uai_keyvault_secrets_user" {
  scope                = var.key_vault_id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

resource "azurerm_role_assignment" "mlw_uai_keyvault_secrets_officer" {
  scope                = var.key_vault_id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

# Grant the user-assigned managed identity access to Container Registry
resource "azurerm_role_assignment" "mlw_uai_acr_pull" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

resource "azurerm_role_assignment" "mlw_uai_acr_push" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPush"
  principal_id         = azurerm_user_assigned_identity.mlw_uai.principal_id
}

# Wait for RBAC propagation - Azure typically needs 60-120 seconds
resource "time_sleep" "wait_for_rbac_propagation" {
  create_duration = "120s"
  
  depends_on = [
    azurerm_role_assignment.mlw_uai_storage_blob_data_reader,
    azurerm_role_assignment.mlw_uai_storage_blob_data_contributor,
    azurerm_role_assignment.mlw_uai_storage_account_contributor,
    azurerm_role_assignment.mlw_uai_keyvault_reader,
    azurerm_role_assignment.mlw_uai_keyvault_secrets_user,
    azurerm_role_assignment.mlw_uai_keyvault_secrets_officer,
    azurerm_role_assignment.mlw_uai_acr_pull,
    azurerm_role_assignment.mlw_uai_acr_push
  ]
}

resource "azurerm_machine_learning_workspace" "mlw" {
  name                    = "mlw-${var.prefix}-${var.postfix}${var.env}"
  location                = var.location
  resource_group_name     = var.rg_name
  application_insights_id = var.application_insights_id
  key_vault_id            = var.key_vault_id
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id

  sku_name                          = "Basic"
  public_network_access_enabled     = true
  image_build_compute_name          = "cpu-cluster"
  v1_legacy_mode_enabled            = false

  identity {
    type         = "SystemAssigned, UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.mlw_uai.id]
  }

  primary_user_assigned_identity = azurerm_user_assigned_identity.mlw_uai.id

  tags = var.tags
  
  # Wait for RBAC permissions to propagate
  depends_on = [
    time_sleep.wait_for_rbac_propagation
  ]
}

# Grant the workspace system-assigned managed identity access to storage account
resource "azurerm_role_assignment" "mlw_system_storage_blob_data_contributor" {
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_workspace.mlw.identity[0].principal_id
}

# Grant the workspace system-assigned managed identity access to Key Vault
resource "azurerm_role_assignment" "mlw_system_keyvault_secrets_officer" {
  scope                = var.key_vault_id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = azurerm_machine_learning_workspace.mlw.identity[0].principal_id
}

# Grant the workspace system-assigned managed identity access to Container Registry
resource "azurerm_role_assignment" "mlw_system_acr_push" {
  scope                = var.container_registry_id
  role_definition_name = "AcrPush"
  principal_id         = azurerm_machine_learning_workspace.mlw.identity[0].principal_id
}

# Grant GitHub Actions service principal access to storage account (for CI/CD pipelines)
resource "azurerm_role_assignment" "github_actions_storage_blob_data_reader" {
  count                = var.github_actions_service_principal_id != "" ? 1 : 0
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = var.github_actions_service_principal_id
}

resource "azurerm_role_assignment" "github_actions_storage_blob_data_contributor" {
  count                = var.github_actions_service_principal_id != "" ? 1 : 0
  scope                = var.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.github_actions_service_principal_id
}

# Compute cluster

resource "azurerm_machine_learning_compute_cluster" "adl_aml_ws_compute_cluster" {
  name                          = "cpu-cluster"
  location                      = var.location
  vm_priority                   = "Dedicated"
  vm_size                       = "STANDARD_D4S_V3"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.mlw.id
  count                         = var.enable_aml_computecluster ? 1 : 0

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.mlw_uai.id]
  }

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 4
    scale_down_nodes_after_idle_duration = "PT120S" # 120 seconds
  }
}

# Private endpoint for ML Workspace
resource "azurerm_private_endpoint" "mlw_pe" {
  count               = var.enable_private_endpoints ? 1 : 0
  name                = "pe-${azurerm_machine_learning_workspace.mlw.name}"
  location            = var.location
  resource_group_name = var.rg_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "psc-${azurerm_machine_learning_workspace.mlw.name}"
    private_connection_resource_id = azurerm_machine_learning_workspace.mlw.id
    subresource_names              = ["amlworkspace"]
    is_manual_connection           = false
  }

  tags = var.tags
}
