// Execute this main file to configure Azure Machine Learning end-to-end in a moderately secure set up
// this template and submodule are from here https://github.com/Azure/azure-quickstart-templates
// Parameters
targetScope='subscription'

@minLength(2)
@maxLength(10)
@description('Prefix for all resource names.')
param nameSpace string

@description('Azure region used for the deployment of all resources.')
param location string

@description('Set of tags to apply to all resources.')
param tags object = {}

@description('Virtual network address prefix')
param vnetAddressPrefix string = '192.168.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '192.168.0.0/24'

@description('Scoring subnet address prefix')
param scoringSubnetPrefix string = '192.168.1.0/24'

@description('Bastion subnet address prefix')
param azureBastionSubnetPrefix string = '192.168.250.0/27'


@description('Deploy a Bastion jumphost to access the network-isolated environment?')
param deployJumphost bool = true

@description('Jumphost virtual machine username')
param dsvmJumpboxUsername string

@secure()
@minLength(8)
@description('Jumphost virtual machine password')
param dsvmJumpboxPassword string

@description('Enable public IP for Azure Machine Learning compute nodes')
param amlComputePublicIp bool = false


// Variables
var name = toLower('${nameSpace}')

resource resgrp 'Microsoft.Resources/resourceGroups@2021-04-01'={
  name: 'rg-${location}-${nameSpace}'
  location: location
  tags: tags
}

// Create a short, unique suffix, that will be unique to each resource group
var uniqueSuffix = substring(uniqueString(resgrp.id), 0, 4)

// Virtual network and network security group
module nsg 'modulessec/nsg.bicep' = { 
  name: 'nsg-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    tags: tags 
    nsgName: 'nsg-${name}-${uniqueSuffix}'
  }
}

module vnet 'modulessec/vnet.bicep' = { 
  name: 'vnet-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    virtualNetworkName: 'vnet-${name}-${uniqueSuffix}'
    networkSecurityGroupId: nsg.outputs.networkSecurityGroup
    vnetAddressPrefix: vnetAddressPrefix
    trainingSubnetPrefix: trainingSubnetPrefix
    scoringSubnetPrefix: scoringSubnetPrefix
    azureBastionSubnetPrefix: azureBastionSubnetPrefix
    tags: tags
  }
}


// Dependent resources for the Azure Machine Learning workspace
module keyvault 'modulessec/keyvault.bicep' = {
  name: 'kv-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    keyvaultName: 'kv-${name}-${uniqueSuffix}'
    keyvaultPleName: 'ple-${name}-${uniqueSuffix}-kv'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module storage 'modulessec/storage.bicep' = {
  name: 'st${name}${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    storageName: 'st${name}${uniqueSuffix}'
    storagePleBlobName: 'ple-${name}-${uniqueSuffix}-st-blob'
    storagePleFileName: 'ple-${name}-${uniqueSuffix}-st-file'
    storageSkuName: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module containerRegistry 'modulessec/containerregistry.bicep' = {
  name: 'cr${name}${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    containerRegistryName: 'cr${name}${uniqueSuffix}'
    containerRegistryPleName: 'ple-${name}-${uniqueSuffix}-cr'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module applicationInsights 'modulessec/applicationinsights.bicep' = {
  name: 'appi-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    applicationInsightsName: 'appi-${name}-${uniqueSuffix}'
    tags: tags
  }
}

module azuremlWorkspace 'modulessec/machinelearning.bicep' = {
  name: 'mlw-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    // workspace organization
    machineLearningName: 'mlw-${name}-${uniqueSuffix}'
    machineLearningFriendlyName: 'Private link endpoint sample workspace'
    machineLearningDescription: 'This is an example workspace having a private link endpoint.'
    location: location
    //prefix: name
    tags: tags

    // dependent resources
    applicationInsightsId: applicationInsights.outputs.applicationInsightsId
    containerRegistryId: containerRegistry.outputs.containerRegistryId
    keyVaultId: keyvault.outputs.keyvaultId
    storageAccountId: storage.outputs.storageId

    // networking
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    computeSubnetId: '${vnet.outputs.id}/subnets/snet-training'
    //aksSubnetId: '${vnet.outputs.id}/subnets/snet-scoring'
    virtualNetworkId: vnet.outputs.id
    machineLearningPleName: 'ple-${name}-${uniqueSuffix}-mlw'

    // compute
    amlComputePublicIp: amlComputePublicIp
 
  }
  dependsOn: [
    keyvault
    containerRegistry
    applicationInsights
    storage
  ]
}

// Optional VM and Bastion jumphost to help access the network isolated environment
module dsvm 'modulessec/dsvmjumpbox.bicep' = if (deployJumphost) {
  name: 'vm-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    location: location
    virtualMachineName: 'vm-${name}-${uniqueSuffix}'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    adminUsername: dsvmJumpboxUsername
    adminPassword: dsvmJumpboxPassword
    networkSecurityGroupId: nsg.outputs.networkSecurityGroup 
  }
}

module bastion 'modulessec/bastion.bicep' = if (deployJumphost) {
  name: 'bas-${name}-${uniqueSuffix}-deployment'
  scope: resgrp
  params: {
    bastionHostName: 'bas-${name}-${uniqueSuffix}'
    location: location
    bastionSubnetId: vnet.outputs.bastionid
  }
  dependsOn: [
    vnet
  ]
}

