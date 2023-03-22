import yaml
import os
import logging
from pathlib import Path
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from feathr import FeathrClient

from tempfile import TemporaryDirectory
# from threading import local
from urllib.parse import urlparse
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

# from feathr.datasets import NYC_TAXI_SMALL_URL
from feathr_utils.dataset_utils import maybe_download
from feathr_utils.utils_platform import is_databricks



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def get_credential():
    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=True)
    # client_id = "c3830013-88b5-402a-b256-d5031e4f3b19"
    # credential = ManagedIdentityCredential(client_id=client_id)
    return credential


def set_required_feathr_config(
        key_vault_name: str,
        synapse_workspace_name: str,
        adls_account: str,
        adls_fs_name: str,
        webapp_name: str,
        # credential: DefaultAzureCredential
        credential
):

    # # Get all the required credentials from Azure Key Vault
    # key_vault_name = "kv-"+resource_prefix+"-"+resource_postfix+resource_env
    # key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    # synapse_workspace_url = "sy"+resource_prefix+"-"+resource_postfix+resource_env
    # adls_account = "st"+resource_prefix+resource_postfix+resource_env
    # adls_fs_name = "dl"+resource_prefix+resource_postfix+resource_env
    # # resource_prefix = config.get("variables").get("namespace")
    # # key_vault_name = resource_prefix + "kv"
    # # synapse_workspace_url = resource_prefix + "syws"
    # # adls_account = resource_prefix + "dls"
    # # adls_fs_name = resource_prefix + "fs"
    # Check if given credential can get token successfully.
    print(type(credential))
    print("Environment variables: {")
    for k, v in os.environ.items():
        print(f"{k}: {v}")
    print("}")
    # print("Attempting to get access token...")
    # print("Access token:", credential.get_token("https://management.azure.com/.default"))
    # print("Got access token!")
    print()

    key_vault_url = f"https://{key_vault_name}.vault.azure.net"
    client = SecretClient(vault_url=key_vault_url, credential=credential)
    secretName = "FEATHR-ONLINE-STORE-CONN"
    retrieved_secret = str(client.get_secret(secretName).value)

    # Get redis credentials; This is to parse Redis connection string.
    redis_port = retrieved_secret.split(',')[0].split(":")[1]
    redis_host = retrieved_secret.split(',')[0].split(":")[0]
    redis_password = retrieved_secret.split(',')[1].split("password=", 1)[1]
    redis_ssl = retrieved_secret.split(',')[2].split("ssl=", 1)[1]

    # Set appropriate environment variables for overriding feathr config
    os.environ['spark_config__azure_synapse__dev_url'] = f'https://{synapse_workspace_name}.dev.azuresynapse.net'
    os.environ['spark_config__azure_synapse__pool_name'] = 'spdev'
    os.environ['spark_config__azure_synapse__workspace_dir'] = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_project'
    os.environ['online_store__redis__host'] = redis_host
    os.environ['online_store__redis__port'] = redis_port
    os.environ['online_store__redis__ssl_enabled'] = redis_ssl
    os.environ['REDIS_PASSWORD'] = redis_password
    os.environ['FEATURE_REGISTRY__API_ENDPOINT'] = f'https://{webapp_name}.azurewebsites.net/api/v1'


def get_feathr_client( 
    key_vault_name: str,
    synapse_workspace_name: str,
    adls_account: str,
    adls_fs_name: str,
    webapp_name: str,
):
    credential = get_credential()
    set_required_feathr_config(key_vault_name=key_vault_name, synapse_workspace_name=synapse_workspace_name, adls_account=adls_account, adls_fs_name=adls_fs_name, webapp_name=webapp_name, credential=credential)
    config_file_path = os.path.join(
        Path(__file__).parent, "feathr_config.yaml")
    logging.info("config path: {}".format(config_file_path))
    return FeathrClient(config_path=config_file_path, credential=credential)
    
# The two methods below are for retrieving raw data in prep.py
def get_pandas_df(
    data_url: str,
    local_cache_path: str = None
) -> pd.DataFrame:
    """Get NYC taxi fare prediction data samples as a pandas DataFrame.
    Refs:
        https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    Args:
        local_cache_path (optional): Local cache file path to download the data set.
            If local_cache_path is a directory, the source file name will be added.
    Returns:
        pandas DataFrame
    """
    # if local_cache_path params is not provided then create a temporary folder
    if local_cache_path is None:
        local_cache_path = TemporaryDirectory().name

    # If local_cache_path is a directory, add the source file name.
    src_filepath = Path(urlparse(data_url).path)
    dst_path = Path(local_cache_path)
    if dst_path.suffix != src_filepath.suffix:
        local_cache_path = str(dst_path.joinpath(src_filepath.name))

    maybe_download(src_url=data_url, dst_filepath=local_cache_path)

    pdf = pd.read_csv(local_cache_path)

    return pdf


def get_spark_df(
    data_url: str,
    spark: SparkSession,
    local_cache_path: str,
) -> DataFrame:
    """Get NYC taxi fare prediction data samples as a spark DataFrame.
    Refs:
        https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    Args:
        spark: Spark session.
        local_cache_path: Local cache file path to download the data set.
            If local_cache_path is a directory, the source file name will be added.
    Returns:
        Spark DataFrame
    """
    # In spark, local_cache_path should be a persist directory or file path
    if local_cache_path is None:
        raise ValueError("In spark, `local_cache_path` should be a persist directory or file path.")

    # If local_cache_path is a directory, add the source file name.
    src_filepath = Path(urlparse(data_url).path)
    dst_path = Path(local_cache_path)
    if dst_path.suffix != src_filepath.suffix:
        local_cache_path = str(dst_path.joinpath(src_filepath.name))

    if is_databricks():
        # Databricks uses "dbfs:/" prefix for spark paths
        if not local_cache_path.startswith("dbfs:"):
            local_cache_path = f"dbfs:/{local_cache_path.lstrip('/')}"
        # Databricks uses "/dbfs/" prefix for python paths
        python_local_cache_path = local_cache_path.replace("dbfs:", "/dbfs")
    # TODO add "if is_synapse()"
    else:
        python_local_cache_path = local_cache_path

    maybe_download(src_url=data_url, dst_filepath=python_local_cache_path)

    df = spark.read.option("header", True).csv(local_cache_path)

    return df

# The three methods below are used for writing/reading data from ADLS storage
def create_or_retrieve_file_system(service_client, file_system):
    '''Creates a new file system on a given service client, or returns an existing one.'''
    file_system_client = service_client.get_file_system_client(file_system)
    if file_system_client.exists():
        print("File system already exists:", file_system)
    else:
        file_system_client.create_file_system()
        print("File system created:", file_system)
    return file_system_client

def create_or_retrieve_directory(file_system_client, directory):
    directory_client = file_system_client.get_directory_client(directory)
    if directory_client.exists():
        print("Directory already exists:", directory)
    else:
        directory_client.create_directory()
        print("Directory created:", directory)
    return directory_client

def create_or_retrieve_file(directory_client, file):
    file_client = directory_client.get_file_client(file)
    if file_client.exists():
        print("File already exists:", file)
    else:
        file_client.create_file()
    return file_client
