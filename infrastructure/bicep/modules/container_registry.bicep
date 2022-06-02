param baseName string
param location string
param tags object

resource cr 'Microsoft.ContainerRegistry/registries@2020-11-01-preview' = {
  name: 'cr${baseName}'
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
