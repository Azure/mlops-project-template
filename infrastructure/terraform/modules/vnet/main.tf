# Virtual Network for Azure Machine Learning
resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  address_space       = [var.vnet_address_space]

  tags = var.tags
}

# Subnet for training compute resources
resource "azurerm_subnet" "training" {
  name                 = "snet-training"
  resource_group_name  = var.rg_name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = [var.training_subnet_address_prefix]

  # Required for private endpoints
  private_endpoint_network_policies             = "Disabled"
  private_link_service_network_policies_enabled = false
}

# Subnet for private endpoints
resource "azurerm_subnet" "endpoints" {
  name                 = "snet-endpoints"
  resource_group_name  = var.rg_name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = [var.endpoints_subnet_address_prefix]

  # Required for private endpoints
  private_endpoint_network_policies             = "Disabled"
  private_link_service_network_policies_enabled = false
}

# Network Security Group for training subnet
resource "azurerm_network_security_group" "training" {
  name                = "nsg-training-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name

  # Allow inbound from Azure Machine Learning service tag
  security_rule {
    name                       = "AllowAzureMachineLearningInbound"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_ranges    = ["44224"]
    source_address_prefix      = "AzureMachineLearning"
    destination_address_prefix = "*"
  }

  # Allow inbound from Batch node management
  security_rule {
    name                       = "AllowBatchNodeManagementInbound"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_ranges    = ["29876-29877"]
    source_address_prefix      = "BatchNodeManagement"
    destination_address_prefix = "*"
  }

  # Allow outbound to Azure Storage
  security_rule {
    name                       = "AllowStorageOutbound"
    priority                   = 100
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "Storage"
  }

  # Allow outbound to Azure Key Vault
  security_rule {
    name                       = "AllowKeyVaultOutbound"
    priority                   = 110
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "AzureKeyVault"
  }

  # Allow outbound to Azure Container Registry
  security_rule {
    name                       = "AllowAcrOutbound"
    priority                   = 120
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "AzureContainerRegistry"
  }

  # Allow outbound to Azure Machine Learning
  security_rule {
    name                       = "AllowAzureMachineLearningOutbound"
    priority                   = 130
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "AzureMachineLearning"
  }

  # Allow outbound to Azure Active Directory
  security_rule {
    name                       = "AllowAzureActiveDirectoryOutbound"
    priority                   = 140
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "AzureActiveDirectory"
  }

  # Allow outbound to Azure Resource Manager
  security_rule {
    name                       = "AllowAzureResourceManagerOutbound"
    priority                   = 150
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "AzureResourceManager"
  }

  # Allow outbound to Azure Monitor
  security_rule {
    name                       = "AllowAzureMonitorOutbound"
    priority                   = 160
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "AzureMonitor"
  }

  tags = var.tags
}

# Associate NSG with training subnet
resource "azurerm_subnet_network_security_group_association" "training" {
  subnet_id                 = azurerm_subnet.training.id
  network_security_group_id = azurerm_network_security_group.training.id
}

# Private DNS zones for Azure services

# Private DNS zone for Azure Machine Learning workspace
resource "azurerm_private_dns_zone" "aml_api" {
  name                = "privatelink.api.azureml.ms"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "aml_api" {
  name                  = "link-aml-api"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.aml_api.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}

# Private DNS zone for Azure Machine Learning notebooks
resource "azurerm_private_dns_zone" "aml_notebooks" {
  name                = "privatelink.notebooks.azure.net"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "aml_notebooks" {
  name                  = "link-aml-notebooks"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.aml_notebooks.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}

# Private DNS zone for Storage Blob
resource "azurerm_private_dns_zone" "blob" {
  name                = "privatelink.blob.core.windows.net"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "blob" {
  name                  = "link-blob"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.blob.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}

# Private DNS zone for Storage File
resource "azurerm_private_dns_zone" "file" {
  name                = "privatelink.file.core.windows.net"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "file" {
  name                  = "link-file"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.file.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}

# Private DNS zone for Storage DFS (Data Lake Gen2)
resource "azurerm_private_dns_zone" "dfs" {
  name                = "privatelink.dfs.core.windows.net"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "dfs" {
  name                  = "link-dfs"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.dfs.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}

# Private DNS zone for Key Vault
resource "azurerm_private_dns_zone" "keyvault" {
  name                = "privatelink.vaultcore.azure.net"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "keyvault" {
  name                  = "link-keyvault"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.keyvault.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}

# Private DNS zone for Container Registry
resource "azurerm_private_dns_zone" "acr" {
  name                = "privatelink.azurecr.io"
  resource_group_name = var.rg_name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "acr" {
  name                  = "link-acr"
  resource_group_name   = var.rg_name
  private_dns_zone_name = azurerm_private_dns_zone.acr.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
  tags                  = var.tags
}
