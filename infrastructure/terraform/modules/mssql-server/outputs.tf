output "mssql_server_name" {
  value = azurerm_mssql_server.mssql_server.name
}

output "mssql_db_name" {
  value = azurerm_mssql_database.mssql_db.name
}
