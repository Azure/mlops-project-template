output "id" {
  value = azurerm_storage_account.st.id
}

output "name" {
  value = azurerm_storage_account.st.name
}

output "filesystem_id" {
  value = var.enable_feature_store == true ? azurerm_storage_data_lake_gen2_filesystem.st_filesystem[0].id : ""
}
