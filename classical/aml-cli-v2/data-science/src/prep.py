# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and uses Feathr feature store to create additional features for further training
"""

from pyspark.sql import DataFrame
from feathr import INPUT_CONTEXT, HdfsSource
from feathr import BOOLEAN, FLOAT, INT32, ValueType
from feathr import Feature, DerivedFeature, FeatureAnchor
from feathr import TypedKey, WindowAggTransformation
from feathr import BackfillTime, MaterializationSettings, RedisSink
from feathr import FeatureQuery, ObservationSettings
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F
import utils

import argparse
import os
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

from azure.identity import AzureCliCredential, DefaultAzureCredential 
from azure.keyvault.secrets import SecretClient

from feathr.spark_provider.feathr_configurations import SparkExecutionConfiguration
# from feathr_utils import nyc_taxi
from feathr_utils.job_utils import get_result_df
from feathr_utils.utils_platform import is_databricks

from azure.storage.filedatalake import DataLakeServiceClient

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--resource_prefix", type=str, help="Resource prefix used for all resource names (not necessarily resource group name).")
    # parser.add_argument("--resource_group", type=str, help="The name of the resource group.")
    # parser.add_argument("--azure_client_id", type=str)
    # parser.add_argument("--azure_tenant_id", type=str)
    # parser.add_argument("--azure_client_secret", type=str)
    # parser.add_argument("--azure_subscription_id", type=str)
    # parser.add_argument("--adls_key", type=str)

    # parser.add_argument("--azure_client_id", type=str)
    # parser.add_argument("--azure_tenant_id", type=str)
    parser.add_argument("--key_vault_name", type=str)
    parser.add_argument("--synapse_workspace_name", type=str)
    parser.add_argument("--adls_account", type=str)
    parser.add_argument("--adls_fs_name", type=str)
    parser.add_argument("--webapp_name", type=str)
    parser.add_argument("--raw_data", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--sp_client_id", type=str)
    parser.add_argument("--sp_client_secret", type=str)
    parser.add_argument("--tenant_id", type=str)
    args = parser.parse_args()

    return args

def set_environment_variables():
    load_dotenv()
    # os.environ['RESOURCE_GROUP'] = utils.fs_config.get("resource_group")
    # os.environ['RESOURCE_PREFIX'] = utils.fs_config.get("resource_prefix")
    # os.environ['AZURE_SUBSCRIPTION_ID'] = utils.fs_config.get("subscription_id")
    # os.environ['AZURE_CLIENT_ID'] = utils.fs_config.get("client_id")
    # os.environ['AZURE_TENANT_ID'] = utils.fs_config.get("tenant_id")

    # # TODO: add client secret, adls key to environment variables
    os.environ['AZURE_CLIENT_ID'] = args.sp_client_id
    os.environ['AZURE_CLIENT_SECRET'] = args.sp_client_secret
    os.environ['AZURE_TENANT_ID'] = args.tenant_id
    os.environ['ADLS_ACCOUNT'] = args.adls_account

def set_spark_session():
    global spark
    spark = (
        SparkSession
        .builder
        .appName("feathr")
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.3.0,io.delta:delta-core_2.12:2.1.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.ui.port", "8080")  # Set ui port other than the default one (4040) so that feathr spark job doesn't fail. 
        .getOrCreate()
    )

def get_data_source_path(feathr_client):
    DATA_STORE_PATH = TemporaryDirectory().name
    DATA_FILE_PATH = str(Path(DATA_STORE_PATH, "nyc_taxi.csv"))
    # Define data source path
    if feathr_client.spark_runtime == "local" or (feathr_client.spark_runtime == "databricks" and is_databricks()):
        # In local mode, we can use the same data path as the source.
        # If the notebook is running on databricks, DATA_FILE_PATH should be already a dbfs path.
        data_source_path = DATA_FILE_PATH
    else:
        # Otherwise, upload the local file to the cloud storage (either dbfs or adls).
        data_source_path = feathr_client.feathr_spark_launcher.upload_or_get_cloud_path(DATA_FILE_PATH) 

    return data_source_path


def main(args):
    import feathr
    print("Feathr version:", feathr.__version__)
    feathr_client = utils.get_feathr_client(key_vault_name=args.key_vault_name, synapse_workspace_name=args.synapse_workspace_name, adls_account=args.adls_account, adls_fs_name=args.adls_fs_name, webapp_name=args.webapp_name)

    set_spark_session()
    data_source_path = get_data_source_path(feathr_client)
    df_raw = utils.get_spark_df(data_url=args.raw_data, spark=spark, local_cache_path=data_source_path)

    TIMESTAMP_COL = "lpep_dropoff_datetime"
    TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"
    


    def feathr_udf_day_calc(df: DataFrame) -> DataFrame:
        df = df.withColumn("fare_amount_cents", col("fare_amount")*100)
        return df


    # define the data source
    batch_source = HdfsSource(name="nycTaxiBatchSource",
                            # path="abfss://{container_name}@{storage_account}.dfs.core.windows.net/nyc_taxi.parquet".format(
                            #     storage_account=utils.fs_config.get("adls_account"), container_name=utils.fs_config.get("data_container_name")),
                            path=data_source_path,
                            event_timestamp_column="lpep_dropoff_datetime",
                            preprocessing=feathr_udf_day_calc,
                            timestamp_format="yyyy-MM-dd HH:mm:ss")


    # define anchor features
    f_trip_distance = Feature(name="f_trip_distance",
                            feature_type=FLOAT, transform="trip_distance")
    f_trip_time_duration = Feature(name="f_trip_time_duration",
                                feature_type=INT32,
                                transform="(to_unix_timestamp(lpep_dropoff_datetime) - to_unix_timestamp(lpep_pickup_datetime))/60")

    features = [
        f_trip_distance,
        f_trip_time_duration,
        Feature(name="f_is_long_trip_distance",
                feature_type=BOOLEAN,
                transform="cast_float(trip_distance)>30"),
        Feature(name="f_day_of_week",
                feature_type=INT32,
                transform="dayofweek(lpep_dropoff_datetime)"),
    ]

    request_anchor = FeatureAnchor(name="request_features",
                                source=INPUT_CONTEXT,
                                features=features)


    # window aggregation features
    location_id = TypedKey(key_column="DOLocationID",
                        key_column_type=ValueType.INT32,
                        description="location id in NYC",
                        full_name="nyc_taxi.location_id")

    
    # calculate the average trip fare, maximum fare and total fare per location for 90 days
    agg_features = [Feature(name="f_location_avg_fare",
                            key=location_id,
                            feature_type=FLOAT,
                            transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                            agg_func="AVG",
                                                            window="90d")),
                    Feature(name="f_location_max_fare",
                            key=location_id,
                            feature_type=FLOAT,
                            transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                            agg_func="MAX",
                                                            window="90d")),
                    Feature(name="f_location_total_fare_cents",
                            key=location_id,
                            feature_type=FLOAT,
                            transform=WindowAggTransformation(agg_expr="fare_amount_cents",
                                                            agg_func="SUM",
                                                            window="90d")),
                    ]

    agg_anchor = FeatureAnchor(name="aggregationFeatures",
                            source=batch_source,
                            features=agg_features)

    # derived features
    f_trip_time_distance = DerivedFeature(name="f_trip_time_distance",
                                        feature_type=FLOAT,
                                        input_features=[
                                            f_trip_distance, f_trip_time_duration],
                                        transform="f_trip_distance * f_trip_time_duration")

    f_trip_time_rounded = DerivedFeature(name="f_trip_time_rounded",
                                        feature_type=INT32,
                                        input_features=[f_trip_time_duration],
                                        transform="f_trip_time_duration % 10")

    feathr_client.build_features(anchor_list=[agg_anchor, request_anchor], derived_feature_list=[
        f_trip_time_distance, f_trip_time_rounded])

    
    derived_features = [f_trip_time_distance, f_trip_time_rounded]
    feature_names = [feature.name for feature in features + agg_features + derived_features]
    DATA_FORMAT = "parquet"
    DATA_STORE_PATH = TemporaryDirectory().name
    offline_features_path = str(Path(DATA_STORE_PATH, "feathr_output", f"features.{DATA_FORMAT}"))
    
    query = FeatureQuery(
        feature_list=feature_names,
        key=location_id,
    )
    
    settings = ObservationSettings(
        observation_path=data_source_path,
        event_timestamp_column=TIMESTAMP_COL,
        timestamp_format=TIMESTAMP_FORMAT,
    )

    feathr_client.get_offline_features(
        observation_settings=settings,
        feature_query=query,
        # For more details, see https://feathr-ai.github.io/feathr/how-to-guides/feathr-job-configuration.html
        execution_configurations=SparkExecutionConfiguration({
            "spark.feathr.outputFormat": DATA_FORMAT,
        }),
        output_path=offline_features_path,
    )

    feathr_client.wait_job_to_finish(timeout_sec=1000)


    # register features
    utils.logging.info("registering features")
    feathr_client.register_features()

    # Materialize features

    # Get the last date from the dataset
    backfill_timestamp = (
        df_raw
        .select(F.to_timestamp(F.col(TIMESTAMP_COL), TIMESTAMP_FORMAT).alias(TIMESTAMP_COL))
        .agg({TIMESTAMP_COL: "max"})
        .collect()[0][0]
    )

    FEATURE_TABLE_NAME = "nycTaxiDemoFeature"

    # Time range to materialize
    backfill_time = BackfillTime(
        start=backfill_timestamp,
        end=backfill_timestamp,
        step=timedelta(days=1),
    )

    # Destinations:
    # For online store,
    redis_sink = RedisSink(table_name=FEATURE_TABLE_NAME)

    settings = MaterializationSettings(
        name=FEATURE_TABLE_NAME + ".job",  # job name
        backfill_time=backfill_time,
        sinks=[redis_sink],  # or adls_sink
        feature_names=[feature.name for feature in agg_features],
    )

    feathr_client.materialize_features(
        settings=settings,
        execution_configurations={"spark.feathr.outputFormat": "parquet"},
    )

    feathr_client.wait_job_to_finish(timeout_sec=5000)

    df = get_result_df(
    spark=spark,
    client=feathr_client,
    data_format=DATA_FORMAT,
    res_url=offline_features_path,
    )

    df_processed = (
    df
    .withColumn("label", F.col("fare_amount").cast("double"))
    .where(F.col("f_trip_time_duration") > 0)
    .fillna(0)
    )
    # Convert the data to Pandas format
    df_pandas = df_processed.toPandas()

    # Remove columns that don't work with our regressor
    column_list = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'store_and_fwd_flag', 'f_is_long_trip_distance']
    for col in column_list:
        del df_pandas[col]

    df_pandas = df_pandas.fillna(0)

    # # Create/retrieve the necessary clients and upload the data

    # adls_system_url = f"{utils.fs_config.get('adls_scheme')}://{utils.fs_config.get('adls_account')}.dfs.core.windows.net"
    # service_client = DataLakeServiceClient(
    #     account_url=adls_system_url, credential=os.environ['ADLS_KEY'])


    # file_system_client = utils.create_or_retrieve_file_system(service_client, utils.fs_config.get('adls_file_system'))
    # directory_client = utils.create_or_retrieve_directory(file_system_client, utils.fs_config.get('adls_data_directory'))
    # file_client = utils.create_or_retrieve_file(directory_client, utils.fs_config.get('adls_data_file'))
    # # Convert Pandas Dataframe to CSV and upload to the specified file
    # output = df_pandas.to_csv (index_label="idx", encoding = "utf-8")
    # file_client.upload_data(output, overwrite=True)

    df_pandas.to_parquet((Path(args.train_data) / "aml.parquet"))


if __name__ == '__main__':

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}"
    ]

    for line in lines:
        print(line)
    
    main(args)