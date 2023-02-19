import argparse

def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser()
    # parser.add_argument("--namespace", type=str, help="Original namespace; no fs at the end, it is added to resource prefix inside the script.")
    parser.add_argument("--key_vault_name", type=str)
    parser.add_argument("--synapse_workspace_url", type=str)
    parser.add_argument("--adls_account", type=str)
    parser.add_argument("--adls_fs_name", type=str)
    parser.add_argument("--redis_name", type=str)
    parser.add_argument("--webapp_name", type=str)
    parser.add_argument("--project_name", type=str, help="Name of Feathr project.")
    parser.add_argument("--spark_cluster", type=str, help="Type of Spark cluster.")
    parser.add_argument("--feathr_config_path", type=str, help="Path to resulting feathr config file.")

    args = parser.parse_args()

    return args

def main(args):

    feathr_config = f"""
    # DO NOT MOVE OR DELETE THIS FILE

    # This file contains the configurations that are used by Feathr
    # All the configurations can be overwritten by environment variables with concatenation of `__` for different layers of this config file.
    # For example, `feathr_runtime_location` for databricks can be overwritten by setting this environment variable:
    # SPARK_CONFIG__DATABRICKS__FEATHR_RUNTIME_LOCATION
    # Another example would be overwriting Redis host with this config: `ONLINE_STORE__REDIS__HOST`
    # For example if you want to override this setting in a shell environment:
    # export ONLINE_STORE__REDIS__HOST=feathrazure.redis.cache.windows.net

    # version of API settings
    api_version: 1
    project_config:
    project_name: "{args.project_name}"
    # Information that are required to be set via environment variables.
    required_environment_variables:
        # the environemnt variables are required to run Feathr
        # Redis password for your online store
        - "REDIS_PASSWORD"
        # Client IDs and client Secret for the service principal. Read the getting started docs on how to get those information.
        - "AZURE_CLIENT_ID"
        - "AZURE_TENANT_ID"
        - "AZURE_CLIENT_SECRET"
    optional_environment_variables:
        # the environemnt variables are optional, however you will need them if you want to use some of the services:
        - ADLS_ACCOUNT
        - ADLS_KEY
        - BLOB_ACCOUNT
        - BLOB_KEY
        - S3_ACCESS_KEY
        - S3_SECRET_KEY
        - JDBC_TABLE
        - JDBC_USER
        - JDBC_PASSWORD
        - KAFKA_SASL_JAAS_CONFIG

    offline_store:
    # paths starts with abfss:// or abfs://
    # ADLS_ACCOUNT and ADLS_KEY should be set in environment variable if this is set to true
    adls:
        adls_enabled: true

    # paths starts with wasb:// or wasbs://
    # BLOB_ACCOUNT and BLOB_KEY should be set in environment variable
    wasb:
        wasb_enabled: false

    # paths starts with s3a://
    # S3_ACCESS_KEY and S3_SECRET_KEY should be set in environment variable
    s3:
        s3_enabled: false
        # S3 endpoint. If you use S3 endpoint, then you need to provide access key and secret key in the environment variable as well.
        s3_endpoint: "s3.amazonaws.com"

    # snowflake endpoint
    # snowflake:
    #   url: "dqllago-ol19457.snowflakecomputing.com"
    #   user: "feathrintegration"
    #   role: "ACCOUNTADMIN"

    # jdbc endpoint
    # jdbc:
    #   jdbc_enabled: true
    #   jdbc_database: "feathrtestdb"
    #   jdbc_table: "feathrtesttable"


    spark_config:
    # choice for spark runtime. Currently support: azure_synapse, databricks
    # The `databricks` configs will be ignored if `azure_synapse` is set and vice versa.
    spark_cluster: {args.spark_cluster}
    # configure number of parts for the spark output for feature generation job
    spark_result_output_parts: "1"

    azure_synapse:
        # dev URL to the synapse cluster. Usually it's `https://yourclustername.dev.azuresynapse.net`
        dev_url: "https://{args.synapse_workspace_url}.dev.azuresynapse.net"
        # name of the sparkpool that you are going to use
        pool_name: "spark31"
        # workspace dir for storing all the required configuration files and the jar resources. All the feature definitions will be uploaded here
        workspace_dir: "abfss://{args.adls_fs_name}@{args.adls_account}.dfs.core.windows.net/{args.project_name}"
        executor_size: "Small"
        executor_num: 1
        # This is the location of the runtime jar for Spark job submission. If you have compiled the runtime yourself, you need to specify this location.
        # Or use wasbs://public@azurefeathrstorage.blob.core.windows.net/feathr-assembly-LATEST.jar so you don't have to compile the runtime yourself
        # Local path, path starting with `http(s)://` or `wasbs://` are supported. If not specified, the latest jar from Maven would be used
        # feathr_runtime_location: "wasbs://public@azurefeathrstorage.blob.core.windows.net/feathr-assembly-LATEST.jar"

    databricks:
        # workspace instance
        workspace_instance_url: 'https://adb-6885802458123232.12.azuredatabricks.net/'
        # config string including run time information, spark version, machine size, etc.
        # the config follows the format in the databricks documentation: https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs#--request-structure-6
        # The fields marked as "FEATHR_FILL_IN" will be managed by Feathr. Other parameters can be customizable. For example, you can customize the node type, spark version, number of workers, instance pools, timeout, etc.
        config_template: '{{"run_name":"FEATHR_FILL_IN","new_cluster":{{"spark_version":"9.1.x-scala2.12","node_type_id":"Standard_D3_v2","num_workers":1,"spark_conf":{{"FEATHR_FILL_IN":"FEATHR_FILL_IN"}}}},"libraries":[{{"jar":"FEATHR_FILL_IN"}}],"spark_jar_task":{{"main_class_name":"FEATHR_FILL_IN","parameters":["FEATHR_FILL_IN"]}}}}'
        # workspace dir for storing all the required configuration files and the jar resources. All the feature definitions will be uploaded here
        work_dir: "dbfs:/{args.project_name}"
        # This is the location of the runtime jar for Spark job submission. If you have compiled the runtime yourself, you need to specify this location.
        # Or use https://azurefeathrstorage.blob.core.windows.net/public/feathr-assembly-LATEST.jar so you don't have to compile the runtime yourself
        # Local path, path starting with `http(s)://` or `dbfs://` are supported. If not specified, the latest jar from Maven would be used
        # feathr_runtime_location: "https://azurefeathrstorage.blob.core.windows.net/public/feathr-assembly-LATEST.jar"

    online_store:
    redis:
        # Redis configs to access Redis cluster
        host: "{args.redis_name}.redis.cache.windows.net"
        port: 6380
        ssl_enabled: True

    feature_registry:
    api_endpoint: "https://{args.webapp_name}.azurewebsites.net/api/v1"
    # # Registry configs if use purview
    # purview:
    #   # configure the name of the purview endpoint
    #   purview_name: <purview-name>
    #   # delimiter indicates that how the project/workspace name, feature names etc. are delimited. By default it will be '__'
    #   # this is for global reference (mainly for feature sharing). For example, when we setup a project called foo, and we have an anchor called 'taxi_driver' and the feature name is called 'f_daily_trips'
    #   # the feature will have a globally unique name called 'foo__taxi_driver__f_daily_trips'
    #   delimiter: "__"
    #   # controls whether the type system will be initialized or not. Usually this is only required to be executed once.
    #   type_system_initialization: false


    secrets:
    azure_key_vault:
        name: {args.key_vault_name}
    """

    with open(args.feathr_config_path, "w") as file:
        file.write(feathr_config)

if __name__ == '__main__':
    args = parse_args()

    main(args)

    print("Feathr config file created:", args.feathr_config_path)