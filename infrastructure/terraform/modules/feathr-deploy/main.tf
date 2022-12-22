data "azurerm_client_config" "current" {}

resource "azurerm_key_vault_secret" "feathr_prefix" {
 count        = var.enable_feathr_deployment ? 1 : 0
 name         = "${var.name_prefix}-kv"
 key_vault_id = azurerm_key_vault
 location            = azurerm_resource_group.name_prefix.value.location
 resource_group_name = azurerm_resource_group.name_prefix.value.name
 tenant_id           = data.azurerm_client_config.current.tenant_id
 sku_name            = "standard"
}


# 
# resource "azurerm_redis_cache" "feathr" {
  # count        = var.enable_feathr_deployment ? 1 : 0
  # name         = "${var.feathr_prefix}-redis"
  # location     = data.azurerm_resource_group.feathr.location
  # resource_group_name = data.azurerm_resource_group.feathr.name
  # capacity     = var.redis_capacity
  # family = "P"
  # sku_name = "Premium"
  # redis_version       = "6"
  # redis_configuration {
    # enable_authentication = false
  # }
  # subnet_id = azurerm_subnet.redis.id
# }

# To include
# location, tenantId, redisCacheName, keyVaultName, keyVault, redisCache, sparkPoolName, 
# workspaceName, dlsName, dlsFsName, dlsAccount, identityName,
# roleDefinitionIdForKeyVaultSecretsUser, roleAssignmentNameForKeyVaultSecretsUser, 
# webAppName, webAppPlanName, webAppPlanSku, webAppAPIVersion, sqlServerName,
#  sqlDatabaseName, sourceBacpacBlobUrl, bacpacBlobName, destinationBacpacBlobUrl, 
#  bacpacDeploymentScriptName, bacpacDbExtensionName, preBuiltdockerImage







