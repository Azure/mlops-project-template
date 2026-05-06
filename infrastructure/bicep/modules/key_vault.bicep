param baseName string
param location string
param tags object
param enablePurgeProtection bool = false
param softDeleteRetentionDays int = 7

// Key Vault — RBAC-authorized, idempotent with soft-delete handling
// Note: enablePurgeProtection cannot be disabled once enabled
resource kv 'Microsoft.KeyVault/vaults@2024-04-01-preview' = {
  name: 'kv-${baseName}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: softDeleteRetentionDays
    enablePurgeProtection: enablePurgeProtection ? true : null
  }

  tags: tags
}

output kvOut string = kv.id
