# Keyless Azure ML Workspace Setup Notes

This document tracks the manual configuration steps required for keyless Azure ML deployment with user-assigned managed identities.

## Infrastructure Overview
- **Resource Group**: rg-ml-82406dev
- **Workspace**: mlw-ml-82406dev
- **Storage Account**: stml82406dev (keyless, no shared access keys)
- **Key Vault**: kv-ml-82406dev (RBAC-enabled)
- **Container Registry**: crml82406dev
- **User-Assigned Managed Identity**: uai-ml-82406dev
  - Principal ID: e57e969f-9334-4107-93f6-7b00d70a4ccb
  - Client ID: 68e20efd-b7ca-44fb-a6ff-3e02f5cf1168
- **Workspace System-Assigned Identity**: c58ad586-4df0-49e2-9fcf-e60a4b97b79d
- **Compute Cluster**: cpu-cluster (STANDARD_D4S_V3, 0-4 nodes)

## Required RBAC Permissions

### User-Assigned Managed Identity (uai-ml-82406dev)
```bash
# Storage Account permissions
az role assignment create --assignee e57e969f-9334-4107-93f6-7b00d70a4ccb \
  --role "Storage Blob Data Reader" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.Storage/storageAccounts/stml82406dev

az role assignment create --assignee e57e969f-9334-4107-93f6-7b00d70a4ccb \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.Storage/storageAccounts/stml82406dev

# Key Vault permissions
az role assignment create --assignee e57e969f-9334-4107-93f6-7b00d70a4ccb \
  --role "Key Vault Secrets Officer" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.KeyVault/vaults/kv-ml-82406dev

# Container Registry permissions
az role assignment create --assignee e57e969f-9334-4107-93f6-7b00d70a4ccb \
  --role "AcrPush" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.ContainerRegistry/registries/crml82406dev

az role assignment create --assignee e57e969f-9334-4107-93f6-7b00d70a4ccb \
  --role "AcrPull" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.ContainerRegistry/registries/crml82406dev
```

### Workspace System-Assigned Identity
```bash
# Storage Account permissions
az role assignment create --assignee c58ad586-4df0-49e2-9fcf-e60a4b97b79d \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.Storage/storageAccounts/stml82406dev

# Key Vault permissions
az role assignment create --assignee c58ad586-4df0-49e2-9fcf-e60a4b97b79d \
  --role "Key Vault Secrets Officer" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.KeyVault/vaults/kv-ml-82406dev

# Container Registry permissions
az role assignment create --assignee c58ad586-4df0-49e2-9fcf-e60a4b97b79d \
  --role "AcrPush" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.ContainerRegistry/registries/crml82406dev
```

### Current User (for manual operations)
```bash
# Storage Account permissions
az role assignment create --assignee $(az ad signed-in-user show --query id -o tsv) \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.Storage/storageAccounts/stml82406dev
```

## Configuration Changes

### 1. Enable Key Vault RBAC Authorization
```bash
az keyvault update --name kv-ml-82406dev \
  --resource-group rg-ml-82406dev \
  --enable-rbac-authorization true
```

### 2. Configure Workspace Image Build Compute
```bash
az ml workspace update --name mlw-ml-82406dev \
  --resource-group rg-ml-82406dev \
  --image-build-compute cpu-cluster
```

### 3. Attach Managed Identity to Compute Cluster
```bash
az ml compute update --name cpu-cluster \
  --resource-group rg-ml-82406dev \
  --workspace-name mlw-ml-82406dev \
  --identity-type user_assigned \
  --user-assigned-identities /subscriptions/c5b4600d-5da3-4298-921b-b2656cfdf1a1/resourceGroups/rg-ml-82406dev/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai-ml-82406dev
```

### 4. Create Workspaceblobstore Datastore
The keyless storage account doesn't auto-provision the default datastore. Created manually:
```bash
az ml datastore create --file classical/aml-cli-v2/mlops/azureml/datastores/workspaceblobstore.yml \
  --resource-group rg-ml-82406dev \
  --workspace-name mlw-ml-82406dev
```

### 5. Create Required Storage Container
```bash
az storage container create \
  --name azureml-blobstore-c5b4600d-5da3-4298-921b-b2656cfdf1a1 \
  --account-name stml82406dev \
  --auth-mode login
```

## Environment Updates
- **Base Image**: Updated from Ubuntu 18.04 to Ubuntu 20.04 (`mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04`)
- **Python**: Updated from 3.7.5 to 3.11
- **Dependencies**: Updated to latest versions:
  - azureml-mlflow: 1.57.0
  - azure-ai-ml: 1.21.1
  - scikit-learn: 1.5.2
  - pandas: 2.2.3
  - pyarrow: 18.1.0

## Known Issues & Solutions

### Issue: Key Vault Permission Denied
**Error**: "The user, group or application does not have secrets set permission on key vault"
**Solution**: Enable RBAC authorization on Key Vault and grant "Key Vault Secrets Officer" role to both user-assigned and system-assigned identities.

### Issue: Container Not Found
**Error**: "The container azureml-blobstore-{subscription-id} does not exist"
**Solution**: Use workspace GUID in container name, not subscription ID. Container format: `azureml-blobstore-{workspace-guid}`

### Issue: Managed Identity Not Found on Compute
**Error**: "Identity of the specified managed compute is not found"
**Solution**: Attach user-assigned managed identity to the compute cluster.

### Issue: ACR Access Denied
**Error**: "denied: requested access to the resource is denied"
**Solution**: Grant "AcrPush" and "AcrPull" roles to both managed identities.

### Issue: Serverless Compute Storage Access
**Error**: "Serverless Computes require access to the workspace's default storage account"
**Solution**: Configure workspace to use cpu-cluster for image builds instead of serverless compute.

## Verification Commands

```bash
# Check workspace configuration
az ml workspace show --name mlw-ml-82406dev --resource-group rg-ml-82406dev

# List datastores
az ml datastore list --resource-group rg-ml-82406dev --workspace-name mlw-ml-82406dev

# Check compute cluster identity
az ml compute show --name cpu-cluster --resource-group rg-ml-82406dev --workspace-name mlw-ml-82406dev --query identity

# List role assignments for managed identity
az role assignment list --assignee e57e969f-9334-4107-93f6-7b00d70a4ccb --output table

# Check Key Vault RBAC setting
az keyvault show --name kv-ml-82406dev --resource-group rg-ml-82406dev --query properties.enableRbacAuthorization
```

## Notes
- RBAC permissions can take 5-10 minutes to fully propagate across Azure services
- Environment images are cached; may need to create new version to force rebuild
- GitHub Actions workflows require OIDC authentication configured in repository settings
