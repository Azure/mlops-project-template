param location string
param prefix string
param postfix string
param workspaceName string

resource amlci 'Microsoft.MachineLearningServices/workspaces/computes@2020-09-01-preview' = {
  name: '${workspaceName}/mlwcc${prefix}${postfix}'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
      }
    }
  }
}
