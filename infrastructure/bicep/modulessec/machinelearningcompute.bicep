// Creates compute resources in the specified machine learning workspace
// Includes Compute Instance, Compute Cluster and attached Azure Kubernetes Service compute types

@description('Azure Machine Learning workspace to create the compute resources in')
param machineLearning string

@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object

@description('Resource ID of the compute subnet')
param computeSubnetId string


@description('Resource ID of the Azure Kubernetes services resource')
param amlComputePublicIp bool


@description('VM size for the default compute cluster')
param amlComputeDefaultVmSize string = 'Standard_Ds3_v2'

resource machineLearningCluster001 'Microsoft.MachineLearningServices/workspaces/computes@2021-07-01' = {
  name: '${machineLearning}/cpu-cluster'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    computeLocation: location
    description: 'Machine Learning cluster 001'
    disableLocalAuth: true
    properties: {
      vmPriority: 'Dedicated'
      vmSize: amlComputeDefaultVmSize
      enableNodePublicIp: amlComputePublicIp
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 2
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      subnet: {
        id: computeSubnetId
      }
    }
  }
}

