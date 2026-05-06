param baseName string
param location string
param stoacctid string
param kvid string
param appinsightid string
param crid string
param tags object

// Optional resource handling — main.bicep may pass empty strings when
// enableMonitoring or enableContainerRegistry are false.
var hasContainerRegistry = !empty(crid)
var hasAppInsights = !empty(appinsightid)

// AML workspace
resource amls 'Microsoft.MachineLearningServices/workspaces@2020-09-01-preview' = {
  name: 'mlw-${baseName}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    tier: 'basic'
    name: 'basic'
  }
  properties: {
    storageAccount: stoacctid
    keyVault: kvid
    applicationInsights: hasAppInsights ? appinsightid : null
    containerRegistry: hasContainerRegistry ? crid : null
    encryption: {
      status: 'Disabled'
      keyVaultProperties: {
        keyIdentifier: ''
        keyVaultArmId: ''
      }
    }
  }

  tags: tags
}

output amlsName string = amls.name
