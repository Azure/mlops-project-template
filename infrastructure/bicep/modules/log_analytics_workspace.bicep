param baseName string
param location string
param tags object

resource log 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name:  'log-${baseName}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
  }
  tags: tags
}

output logOut string = log.id
