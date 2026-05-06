output "name" {
  value = azurerm_machine_learning_workspace.mlw.name
}

output "user_assigned_identity_id" {
  value       = azurerm_user_assigned_identity.mlw_uai.id
  description = "The ID of the user-assigned managed identity for the ML workspace"
}

output "user_assigned_identity_principal_id" {
  value       = azurerm_user_assigned_identity.mlw_uai.principal_id
  description = "The principal ID of the user-assigned managed identity"
}

output "user_assigned_identity_client_id" {
  value       = azurerm_user_assigned_identity.mlw_uai.client_id
  description = "The client ID of the user-assigned managed identity"
}