## Semi-manual setup for feature store components as some AD steps cannot be fully automated
### Create AAD application for feathr UI:

Adjust the values of the variables from config-infra-dev.yml and config-infra-prod.yml, read comments below for detailed instructions

````bash
 This is the prefix you want to name your resources with, make a note of it, you will need it during deployment.
# please keep the following strings short, they are used for resource names and they have a limit of 24 characters on some azure resources. make sure that combined length of the following strings is less than 18 characters.
# make sure to reflect these values from config-infra-dev.yml and config-infra-prod.yml
prefix="mlopsv2fs" # equivalent var in config-infra-*.yml is "namespace" then add "fs" after it (used in feature store nameing internally)
environment="dev" # equivalent var in config-infra-*.yml is "environment"
postfix="001" # equivalent var in config-infra-*.yml is "postfix"

# Please don't change this name, a corresponding webapp with same name gets created in subsequent steps.
sitename="app${prefix}${postfix}${environment}"

# Use the following configuration command to enable dynamic install of az extensions without a prompt. This is required for the az account command group used in the following steps.
az config set extension.use_dynamic_install=yes_without_prompt

# This will create the Azure AD application, note that we need to create an AAD app of platform type Single Page Application(SPA). By default passing the redirect-uris with create command creates an app of type web. Setting Sign in audience to AzureADMyOrg limits the application access to just your tenant.
az ad app create --display-name $sitename --sign-in-audience AzureADMyOrg --web-home-page-url "https://$sitename.azurewebsites.net" --enable-id-token-issuance true
````

### Add redirect URI for the AAD application created above (Feathr UI)):

``
````bash
# Fetch the ClientId, TenantId and ObjectId for the created app
aad_clientId=$(az ad app list --display-name $sitename --query "[].appId" -o tsv)

# We just use the homeTenantId since a user could have access to multiple tenants
aad_tenantId=$(az account show --query "[homeTenantId]" -o tsv)

#Fetch the objectId of AAD app to patch it and add redirect URI in next step.
aad_objectId=$(az ad app list --display-name $sitename --query "[].id" -o tsv)

# Make sure the above command ran successfully and the values are not empty. If they are empty, re-run the above commands as the app creation could take some time.
# MAKE NOTE OF THE CLIENT_ID & TENANT_ID FOR STEP #2
echo "AZURE_AAD_OBJECT_ID: $aad_objectId"
echo "AAD_CLIENT_ID: $aad_clientId" # this should be used in config-infra-*.yml as "aad_client_id"
echo "AZURE_TENANT_ID: $aad_tenantId"

# Updating the SPA app created above, currently there is no CLI support to add redirectUris to a SPA, so we have to patch manually via az rest
az rest --method PATCH --uri "https://graph.microsoft.com/v1.0/applications/$aad_objectId" --headers "Content-Type=application/json" --body "{spa:{redirectUris:['https://$sitename.azurewebsites.net']}}"
````