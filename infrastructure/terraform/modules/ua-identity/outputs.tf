output "uaid_client_id" {
  value = azurerm_user_assigned_identity.ua_identity.client_id
}

output "uaid_id" {
  value = azurerm_user_assigned_identity.ua_identity.id
}

output "uaid_principal_id" {
  value = azurerm_user_assigned_identity.ua_identity.principal_id
}

output "uaid_tenant_id" {
  value = azurerm_user_assigned_identity.ua_identity.tenant_id
}
