# Azure ML VNet and Private Endpoints Implementation

## Overview

This implementation adds comprehensive network isolation support for Azure Machine Learning workspaces and all dependent services using Azure Virtual Networks (VNet) and Private Endpoints. The implementation follows Azure ML best practices and Microsoft's security recommendations.

## Architecture

### Network Topology

```
┌─────────────────────────────────────────────────────────────┐
│ Virtual Network (10.0.0.0/16)                                │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Training Subnet (10.0.0.0/24)                         │  │
│  │ - Azure ML Compute Clusters                           │  │
│  │ - Compute Instances                                    │  │
│  │ - Attached with NSG for Azure ML                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Endpoints Subnet (10.0.1.0/24)                        │  │
│  │ - Private Endpoint: ML Workspace                       │  │
│  │ - Private Endpoint: Storage (blob, file, dfs)         │  │
│  │ - Private Endpoint: Key Vault                          │  │
│  │ - Private Endpoint: Container Registry                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Private DNS Zones │
                    ├──────────────────┤
                    │ privatelink.api.azureml.ms
                    │ privatelink.notebooks.azure.net
                    │ privatelink.blob.core.windows.net
                    │ privatelink.file.core.windows.net
                    │ privatelink.dfs.core.windows.net
                    │ privatelink.vaultcore.azure.net
                    │ privatelink.azurecr.io
                    └──────────────────┘
```

### Components

#### 1. Virtual Network
- **Address Space**: Configurable (default: 10.0.0.0/16)
- **Purpose**: Provides network isolation boundary for all Azure ML resources
- **Subnet Configuration**:
  - **Training Subnet**: For compute cluster nodes and instances
  - **Endpoints Subnet**: For all private endpoint network interfaces

#### 2. Network Security Group (NSG)
Applied to the training subnet with the following rules:

**Inbound Rules**:
- Allow Azure Machine Learning service tag on port 44224
- Allow Batch Node Management on ports 29876-29877

**Outbound Rules**:
- Allow HTTPS (443) to Storage service tag
- Allow HTTPS (443) to AzureKeyVault service tag
- Allow HTTPS (443) to AzureContainerRegistry service tag
- Allow HTTPS (443) to AzureMachineLearning service tag
- Allow HTTPS (443) to AzureActiveDirectory service tag
- Allow HTTPS (443) to AzureResourceManager service tag
- Allow HTTPS (443) to AzureMonitor service tag

#### 3. Private Endpoints
Each service has one or more private endpoints connected to the endpoints subnet:

| Service | Private Endpoints | DNS Zone |
|---------|------------------|----------|
| ML Workspace | 1 (amlworkspace) | privatelink.api.azureml.ms<br>privatelink.notebooks.azure.net |
| Storage Account | 3 (blob, file, dfs*) | privatelink.blob.core.windows.net<br>privatelink.file.core.windows.net<br>privatelink.dfs.core.windows.net |
| Key Vault | 1 (vault) | privatelink.vaultcore.azure.net |
| Container Registry | 1 (registry) | privatelink.azurecr.io |

*DFS endpoint only created if HNS (Hierarchical Namespace) is enabled

#### 4. Private DNS Zones
- **Purpose**: Enable name resolution for private endpoints within the VNet
- **Configuration**: Each DNS zone is linked to the VNet with registration disabled
- **Auto-Registration**: Private DNS zone groups automatically create A records when private endpoints are created

## Configuration

### Enable Private Endpoints

Set the following in `terraform.tfvars`:

```hcl
enable_private_endpoints = true
```

### Customize VNet Address Space

Default configuration provides:
- 65,536 IP addresses total
- 254 addresses for training subnet (compute nodes)
- 254 addresses for endpoints subnet

To customize:

```hcl
vnet_address_space               = "10.0.0.0/16"     # Total address space
training_subnet_address_prefix   = "10.0.0.0/24"    # Compute subnet (254 hosts)
endpoints_subnet_address_prefix  = "10.0.1.0/24"    # Endpoints subnet (254 hosts)
```

### IP Address Planning

**Minimum Requirements**:
- Private endpoints: ~20 IP addresses (one per private endpoint)
- Compute cluster: 1 IP per node (configured max: 4 nodes)
- Compute instances: 1 IP per instance

**Recommended Sizing**:
- Small deployment: /24 subnets (254 addresses each)
- Medium deployment: /23 subnets (510 addresses each)
- Large deployment: /22 subnets (1,022 addresses each)

## Public Network Access

When private endpoints are enabled:
- **ML Workspace**: `public_network_access_enabled = true` (for Azure Portal/Studio access)
- **Storage Account**: `public_network_access_enabled = false`, `default_action = "Deny"`
- **Key Vault**: `default_action = "Deny"` with firewall enabled
- **Container Registry**: `public_network_access_enabled = false`

All services allow access from:
- Azure trusted services (firewall bypass)
- Training subnet via VNet integration
- Private endpoints

## Security Features

### Network Isolation
- All Azure ML resources communicate through private IPs within the VNet
- No data traverses the public internet
- DNS resolution handled by private DNS zones

### Firewall Configuration
- Storage Account: Denies public access, allows training subnet
- Key Vault: Denies public access, allows training subnet, bypasses Azure services
- Container Registry: Denies public access, allows training subnet

### Identity and Access
- Managed identities used for authentication (no keys)
- RBAC roles assigned to workspace and user-assigned identity
- GitHub Actions service principal granted necessary roles

## Deployment

### Initial Deployment with VNet

1. Update `terraform.tfvars`:
   ```hcl
   enable_private_endpoints = true
   ```

2. Deploy infrastructure:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform plan
   terraform apply
   ```

3. Deployment time: ~15-20 minutes (includes VNet, private endpoints, DNS zones)

### Upgrading Existing Deployment

⚠️ **Warning**: Enabling private endpoints on an existing deployment will:
- Modify network configuration of existing resources
- May temporarily affect connectivity during transition
- Require DNS configuration updates for clients

**Recommended Approach**:
1. Test in a new environment first (different postfix)
2. Plan for maintenance window
3. Update `enable_private_endpoints = true`
4. Run `terraform plan` to review changes
5. Apply during maintenance window

## Testing

### Validate VNet Configuration

1. **Check VNet and Subnets**:
   ```bash
   az network vnet show -g <resource-group> -n vnet-<prefix>-<postfix><env> -o table
   az network vnet subnet list -g <resource-group> --vnet-name vnet-<prefix>-<postfix><env> -o table
   ```

2. **Verify Private Endpoints**:
   ```bash
   az network private-endpoint list -g <resource-group> -o table
   ```

3. **Check Private DNS Zones**:
   ```bash
   az network private-dns zone list -g <resource-group> -o table
   ```

### Validate Training Pipeline

1. Run the training pipeline:
   ```bash
   # Via GitHub Actions or locally
   az ml job create --file classical/aml-cli-v2/data-science/src/training_job.yml
   ```

2. Monitor compute cluster:
   - Should provision nodes in training subnet
   - Should access storage/ACR through private endpoints
   - Check logs for connectivity issues

### Validate Endpoint Deployment

1. Deploy batch endpoint:
   ```bash
   az ml batch-endpoint create --file classical/aml-cli-v2/mlops/azureml/deploy/batch/batch-endpoint.yml
   ```

2. Deploy online endpoint:
   ```bash
   az ml online-endpoint create --file classical/aml-cli-v2/mlops/azureml/deploy/online/online-endpoint.yml
   ```

3. Test inference:
   ```bash
   az ml online-endpoint invoke --name <endpoint-name> --request-file test-request.json
   ```

## Troubleshooting

### DNS Resolution Issues

**Symptom**: Cannot resolve privatelink FQDN
**Solution**: 
- Verify private DNS zones are linked to VNet
- Check private DNS zone groups on private endpoints
- Ensure client is connected to VNet (via VPN, ExpressRoute, or jump box)

### Connectivity Timeouts

**Symptom**: Training jobs timeout connecting to storage/ACR
**Solution**:
- Verify NSG rules allow required traffic
- Check firewall rules on storage/ACR allow training subnet
- Ensure "Allow trusted Microsoft services" is enabled

### Studio Access Issues

**Symptom**: Azure ML Studio doesn't load or shows errors
**Solution**:
- Workspace has `public_network_access_enabled = true` for Studio
- Connect via jump box/VPN if Studio features require VNet access
- Check browser can resolve privatelink DNS names

### Private Endpoint Creation Failures

**Symptom**: Terraform fails creating private endpoint
**Solution**:
- Verify subnet has `private_endpoint_network_policies = "Disabled"`
- Check service supports private endpoints (Premium tier for ACR)
- Ensure sufficient IP addresses in endpoints subnet

## Cost Considerations

Private endpoints incur additional costs:
- **Private Endpoint**: ~$0.01 per hour per endpoint (~$7.30/month)
- **Data Processing**: ~$0.01 per GB processed
- **VNet**: No charge for VNet itself

**Total Additional Cost** (approximate):
- 7 private endpoints × $7.30 = ~$51.10/month
- Plus data processing charges based on usage

## Backward Compatibility

The VNet implementation is **fully backward compatible**:
- `enable_private_endpoints` defaults to `false`
- Existing deployments continue working without changes
- VNet module only instantiated when `enable_private_endpoints = true`
- All private endpoint resources are conditional

To maintain current behavior, no action is required.

## Best Practices

1. **Use VNet for Production**: Enable private endpoints for production environments
2. **Plan Address Space**: Ensure address space supports growth
3. **Test First**: Validate VNet config in dev before production
4. **Monitor Connectivity**: Use Azure Monitor to track private endpoint connections
5. **Document DNS**: Keep record of DNS zones and resolution paths
6. **Use Jump Box**: Deploy jump box in VNet for administrative access
7. **Consider VPN**: Set up VPN Gateway for on-premises connectivity

## Related Documentation

- [Azure ML Network Isolation](https://learn.microsoft.com/azure/machine-learning/how-to-network-security-overview)
- [Configure Private Endpoints](https://learn.microsoft.com/azure/machine-learning/how-to-configure-private-link)
- [Secure Azure Storage](https://learn.microsoft.com/azure/storage/common/storage-network-security)
- [Key Vault Private Link](https://learn.microsoft.com/azure/key-vault/general/private-link-service)
- [Container Registry Private Link](https://learn.microsoft.com/azure/container-registry/container-registry-private-link)

## Support

For issues or questions:
1. Check [troubleshooting section](#troubleshooting)
2. Review Azure ML documentation
3. Open GitHub issue with detailed description
4. Include Terraform plan output if relevant
