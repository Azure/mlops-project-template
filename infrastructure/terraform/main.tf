terraform {
  # backend "azurerm" {}  # Commented out for local deployment - uses local state
  required_version = ">= 1.10.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.52.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7.2"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.9.0"
    }
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  use_oidc            = true
  storage_use_azuread = true  # Use Azure AD authentication for storage operations instead of keys
}

data "azurerm_client_config" "current" {}

data "http" "ip" {
  url = "https://ifconfig.me"
}
