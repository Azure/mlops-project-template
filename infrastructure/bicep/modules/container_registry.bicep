param prefix string
param postfix string
param location string
param tags object

resource cr 'Microsoft.ContainerRegistry/registries@2020-11-01-preview' = {
  name: 'cr${prefix}${postfix}'
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
