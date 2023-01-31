data "http" "ip" {
  url = "http://ipv4.icanhazip.com"
}
resource "azurerm_mssql_server" "mssql_server" {
  name                         = "sql${var.prefix}${var.postfix}${var.env}"
  resource_group_name          = var.rg_name
  location                     = var.location
  version                      = "12.0"
  administrator_login          = var.sql_admin_user
  administrator_login_password = var.sql_admin_password
  minimum_tls_version          = "1.2"

  tags = var.tags
}

resource "azurerm_mssql_database" "mssql_db" {
  name           = "db${var.prefix}${var.postfix}${var.env}"
  server_id      = azurerm_mssql_server.mssql_server.id
  collation      = "SQL_Latin1_General_CP1_CI_AS"
  license_type   = "LicenseIncluded"
  max_size_gb    = 2
  read_scale     = false
  sku_name       = "S0"
  zone_redundant = false

  tags = var.tags
}

# this is the current way to allow Azure internal IP to access the SQL server, update when necessary: https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/mssql_firewall_rule
resource "azurerm_mssql_firewall_rule" "allow_azure_internal" {
  name             = "Allow Azure Internal"
  server_id        = azurerm_mssql_server.mssql_server.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# required to create the userroles table in SQL for feature store, should be pushed to the app for creation
# for now as the app does not create this, remove later: https://github.com/feathr-ai/feathr/tree/2cf23a510ff1ed523026b81eabde60b35456b8af/registry/access_control#initialize-userroles-records
# remove steps below later
resource "azurerm_mssql_firewall_rule" "allow_deployment_vm" {
  name             = "AllowDeploymetVM"
  server_id        = azurerm_mssql_server.mssql_server.id
  start_ip_address = chomp(data.http.ip.body)
  end_ip_address   = chomp(data.http.ip.body)
}

# introducing as firewall rule needs time to take effect
resource "time_sleep" "wait_60_seconds" {
  depends_on = [azurerm_mssql_firewall_rule.allow_deployment_vm, azurerm_mssql_database.mssql_db]

  create_duration = "60s"
  # apply the create table for user roles, entities and edges using mssql cli
  provisioner "local-exec" {
    command = "mssql-cli -S ${azurerm_mssql_server.mssql_server.fully_qualified_domain_name} -U ${var.sql_admin_user} -P ${var.sql_admin_password} -d ${azurerm_mssql_database.mssql_db.name} --input_file ./modules/mssql-server/sql/create_userroles.sql"
  }
  provisioner "local-exec" {
    command = "mssql-cli -S ${azurerm_mssql_server.mssql_server.fully_qualified_domain_name} -U ${var.sql_admin_user} -P ${var.sql_admin_password} -d ${azurerm_mssql_database.mssql_db.name} --input_file ./modules/mssql-server/sql/create_entities_edges.sql"
  }
}


