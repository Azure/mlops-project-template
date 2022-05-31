param prefix string
param postfix string
param env string
param location string
param tags object

resource cr 'Microsoft.ContainerRegistry/registries@2020-11-01-preview' = {
  name: 'cr${prefix}${postfix}${env}'
  location: location
  sku: {
    name: 'Standard'
  }

  properties: {
    adminUserEnabled: true
  }

  tags: tags
}

output crOut string = cr.id
