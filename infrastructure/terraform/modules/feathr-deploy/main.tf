data "azurerm_client_config" "current" {}

resource "azurerm_key_vault_secret" "feathr_prefix" {
 count        = var.enable_feathr_deployment ? 1 : 0
 name         = "${var.prefix}-kv-${random_string.postfix.result}"
 key_vault_id = azurerm_key_vault
 location            = azurerm_resource_group.feathr_prefix.value.location
 resource_group_name = azurerm_resource_group.feathr_prefix.value.name
 tenant_id           = data.azurerm_client_config.current.tenant_id
 sku_name            = "standard"
}



resource "azurerm_redis_cache" "feathr" {
  count        = var.enable_feathr_deployment ? 1 : 0
  name         = "${var.feathr_prefix}-redis"
  location     = data.azurerm_resource_group.feathr.location
  resource_group_name = data.azurerm_resource_group.feathr.name
  capacity     = var.redis_capacity
  family = "P"
  sku_name = "Premium"
  redis_version       = "6"
  redis_configuration {
    enable_authentication = false
  }
  subnet_id = azurerm_subnet.redis.id
}

# To include
# location, tenantId, redisCacheName, keyVaultName, keyVault, redisCache, sparkPoolName, 
# workspaceName, dlsName, dlsFsName, dlsAccount, identityName,
# roleDefinitionIdForKeyVaultSecretsUser, roleAssignmentNameForKeyVaultSecretsUser, 
# webAppName, webAppPlanName, webAppPlanSku, webAppAPIVersion, sqlServerName,
#  sqlDatabaseName, sourceBacpacBlobUrl, bacpacBlobName, destinationBacpacBlobUrl, 
#  bacpacDeploymentScriptName, bacpacDbExtensionName, preBuiltdockerImage
#  
resource "azurerm_key_vault_secret" "feathr_online_store_connection" {
 

resource "azurerm_key_vault_secret" "feathr_redis_password" {

    
resource "azurerm_storage_container" "feathr" {

resource "azurerm_key_vault_secret" "feathr_sql_password" {

resource "azurerm_mssql_server" "feathr" {

# if used mssql no need for Purview --> still need it dont consider Purview atm
resource "azurerm_mssql_database" "feathr" {


resource "azurerm_service_plan" "feathr" {

resource "azurerm_user_assigned_identity" "feathr" {

application_stack {
  docker_image     = "feathrfeaturestore/feathr-registry"
  docker_image_tag = "releases-v0.9.0"
}

app_settings = {
 "DOCKER_REGISTRY_SERVER_URL" = "https://index.docker.io/v1"
 "REACT_APP_AZURE_CLIENT_ID"  = var.feathr_webapp_client_id
 "REACT_APP_AZURE_TENANT_ID"  = data.azurerm_client_config.current.tenant_id
 "API_BASE"                   = "api/v1"
 "CONNECTION_STR"             = "Server=tcp:${azurerm_mssql_server.feathr.fully_qualified_dom
 "REACT_APP_ENABLE_RBAC"      = true
 "PURVIEW_NAME"               = azurerm_purview_account.##
 "AZURE_CLIENT_ID"            = azurerm_user_assigned_identity.feathr.client_id
  }

  resource "azurerm_role_assignment" "feathr" {

#private endpoints??? ---> no

resource "azurerm_private_endpoint" "feathr_mssql" {

resource "azurerm_private_endpoint" "feathr_web_app" {


resource "azurerm_private_endpoint" "feathr_redis" {


## private dns zone?--> no
    
resource "azurerm_private_dns_zone_virtual_network_link" "feathr_mssql" {