targetScope = 'subscription'

param location string = 'westus2'
param prefix string
param postfix string
param env string 

// Feature flags — control which optional modules are deployed
param enableMonitoring bool = true
param enableContainerRegistry bool = true
param enableComputeCluster bool = true

// Key Vault settings
param kvEnablePurgeProtection bool = false
param kvSoftDeleteRetentionDays int = 7

// Tag parameters
param tagCostCenter string = ''
param tagManagedBy string = 'bicep'

param tags object = {
  Owner: 'mlops-v2'
  Project: prefix
  Environment: env
  Toolkit: 'bicep'
  Name: prefix
  CostCenter: tagCostCenter
  ManagedBy: tagManagedBy
}

var baseName  = '${prefix}-${postfix}${env}'
var resourceGroupName = 'rg-${baseName}'

resource rg 'Microsoft.Resources/resourceGroups@2020-06-01' = {
  name: resourceGroupName
  location: location

  tags: tags
}

// Storage Account
module st './modules/storage_account.bicep' = {
  name: 'st'
  scope: resourceGroup(rg.name)
  params: {
    baseName: '${uniqueString(rg.id)}${env}'
    location: location
    tags: tags
  }
}

// Key Vault
module kv './modules/key_vault.bicep' = {
  name: 'kv'
  scope: resourceGroup(rg.name)
  params: {
    baseName: baseName
    location: location
    tags: tags
    enablePurgeProtection: kvEnablePurgeProtection
    softDeleteRetentionDays: kvSoftDeleteRetentionDays
  }
}

// App Insights — conditional on enableMonitoring
module appi './modules/application_insights.bicep' = if (enableMonitoring) {
  name: 'appi'
  scope: resourceGroup(rg.name)
  params: {
    baseName: baseName
    location: location
    tags: tags
  }
}

// Container Registry — conditional on enableContainerRegistry
module cr './modules/container_registry.bicep' = if (enableContainerRegistry) {
  name: 'cr'
  scope: resourceGroup(rg.name)
  params: {
    baseName: '${uniqueString(rg.id)}${env}'
    location: location
    tags: tags
  }
}

// AML workspace
module mlw './modules/aml_workspace.bicep' = {
  name: 'mlw'
  scope: resourceGroup(rg.name)
  params: {
    baseName: baseName
    location: location
    stoacctid: st.outputs.stoacctOut
    kvid: kv.outputs.kvOut
    appinsightid: enableMonitoring ? appi!.outputs.appinsightOut : ''
    crid: enableContainerRegistry ? cr!.outputs.crOut : ''
    tags: tags
  }
}

// AML compute cluster — conditional on enableComputeCluster
module mlwcc './modules/aml_computecluster.bicep' = if (enableComputeCluster) {
  name: 'mlwcc'
  scope: resourceGroup(rg.name)
  params: {
    location: location
    workspaceName: mlw.outputs.amlsName
  }
}
