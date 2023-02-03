
locals {
  safe_prefix  = replace(var.prefix, "-", "")
  safe_postfix = replace(var.postfix, "-", "")
  conn_string  = "Server=tcp:${var.mssql_server_name}.database.windows.net,1433;Initial Catalog=${var.mssql_db_name};Persist Security Info=False;User ID=${var.sql_admin_user};Password=${var.sql_admin_password};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"
}

resource "azurerm_storage_account" "stlog" {
  name                     = "stlog${local.safe_prefix}${local.safe_postfix}${var.env}"
  resource_group_name      = var.rg_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = false

  tags = var.tags
}

data "azurerm_storage_account_sas" "stlog_sas" {
  connection_string = azurerm_storage_account.stlog.primary_connection_string
  https_only        = true
  signed_version    = "2017-07-29"

  resource_types {
    service   = true
    container = false
    object    = false
  }

  services {
    blob  = true
    queue = false
    table = false
    file  = false
  }

  # set expiry in a year
  start  = timestamp()
  expiry = timeadd(timestamp(), "8760h")

  permissions {
    read    = true
    write   = true
    delete  = true
    list    = true
    add     = false
    create  = false
    update  = false
    process = false
  }
}

resource "azurerm_service_plan" "app_service_plan" {
  name                = "plan${var.prefix}${var.postfix}${var.env}"
  resource_group_name = var.rg_name
  location            = var.location
  os_type             = "Linux"
  sku_name            = "P1v2"

  tags = var.tags
}

resource "azurerm_linux_web_app" "linux_web_app" {
  name                = "app${var.prefix}${var.postfix}${var.env}"
  resource_group_name = var.rg_name
  location            = var.location
  service_plan_id     = azurerm_service_plan.app_service_plan.id

  site_config {

    application_stack {
      docker_image     = var.feathr_app_image
      docker_image_tag = var.feathr_app_image_tag
    }

  }

  logs {
    detailed_error_messages = true
    failed_request_tracing  = true
    http_logs {
      azure_blob_storage {
        retention_in_days = 15
        sas_url           = data.azurerm_storage_account_sas.stlog_sas.sas
      }

    }
    application_logs {
      file_system_level = "Verbose"
      azure_blob_storage {
        level             = "Verbose"
        retention_in_days = 15
        sas_url           = data.azurerm_storage_account_sas.stlog_sas.sas
      }
    }
  }

  app_settings = {
    DOCKER_REGISTRY_SERVER_URL                 = "https://index.docker.io/v1"
    REACT_APP_AZURE_CLIENT_ID                  = var.aad_client_id
    REACT_APP_AZURE_TENANT_ID                  = var.tenant_id
    API_BASE                                   = "api/v1"
    REACT_APP_ENABLE_RBAC                      = tostring(var.react_enable_rbac)
    AZURE_CLIENT_ID                            = var.uaid_client_id
    DOCKER_ENABLE_CI                           = "true"
    CONNECTION_STR                             = local.conn_string
    APPINSIGHTS_INSTRUMENTATIONKEY             =  var.appi_instrumentation_key
    APPLICATIONINSIGHTS_CONNECTION_STRING      =  var.appi_connection_string
    ApplicationInsightsAgent_EXTENSION_VERSION = "~3"
  }

  tags = var.tags
}
