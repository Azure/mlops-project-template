output "id" {
  value = azurerm_kusto_cluster.cluster[0].id
}

output "uri" {
  value = azurerm_kusto_cluster.cluster[0].uri
}

output "name" {
  value = azurerm_kusto_database.database[0].name
}
