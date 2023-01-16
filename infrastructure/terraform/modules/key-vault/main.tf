data "azurerm_client_config" "current" {}

data "azuread_client_config" "current" {}

resource "azurerm_key_vault" "kv" {
  name                = "kv-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  tags = var.tags

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azuread_client_config.current.object_id

    key_permissions = [
      "Create",
      "Get",
    ]

    secret_permissions = [
      "Set",
      "Get",
      "Delete",
      "Purge",
      "Recover"
    ]
  }
}

# add feature store connection secret if Feature Flag is enabled
resource "azurerm_key_vault_secret" "fs_onlinestore_conn" {
  # Deploy conditionally based on Feature Flag variable
  count = var.enable_feature_store == true ? 1 : 0

  name         = var.fs_onlinestore_conn_name
  value        = var.fs_onlinestore_conn
  key_vault_id = azurerm_key_vault.kv.id

}

# access policy for user assigned identity if Feature Flag is enabled

resource "azurerm_key_vault_access_policy" "fsid_access_policy" {
  # Deploy conditionally based on Feature Flag variable
  count = var.enable_feature_store == true ? 1 : 0

  key_vault_id = azurerm_key_vault.kv.id
  tenant_id    = var.uaid_tenant_id
  object_id    = var.uaid_principal_id

  key_permissions = [
    "Create",
    "Get",
  ]

  secret_permissions = [
    "Set",
    "Get",
    "Delete",
    "Purge",
    "Recover"
  ]
}

# access policy for priviledged user if Feature Flag is enabled
resource "azurerm_key_vault_access_policy" "user_access_policy" {
  # Add policy for user assigned identity if variable is not empty or set
  count = var.priviledged_object_id != "" ? 1 : 0

  key_vault_id = azurerm_key_vault.kv.id
  tenant_id    = var.uaid_tenant_id
  object_id    = var.priviledged_object_id

  key_permissions = [
    "Create",
    "Get",
  ]

  secret_permissions = [
    "Set",
    "Get",
    "Delete",
    "Purge",
    "Recover"
  ]
}
