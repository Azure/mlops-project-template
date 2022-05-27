param prefix string
param postfix string
param location string
param tags object

// App Insights
resource appinsight 'Microsoft.Insights/components@2020-02-02-preview' = {
  name: 'appi-${prefix}-${postfix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }

  tags: tags
}

output appinsightOut string = appinsight.id
