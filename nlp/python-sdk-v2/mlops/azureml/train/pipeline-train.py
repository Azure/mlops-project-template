"""MLOps v2 NLP Python SDK training submission script."""
import os
import argparse

# Azure ML sdk v2 imports
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import dsl, Input, Output

def get_config_parger(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for the script."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    group = parser.add_argument_group("Azure ML references")
    group.add_argument(
        "--config_location",
        type=str,
        required=False,
        help="Subscription ID",
    )
    group.add_argument(
        "--subscription_id",
        type=str,
        required=False,
        help="Subscription ID",
    )
    group.add_argument(
        "--resource_group",
        type=str,
        required=False,
        help="Resource group name",
    )
    group.add_argument(
        "--workspace_name",
        type=str,
        required=False,
        help="Workspace name",
    )
    # Experiment Name
    group.add_argument(
        "-n",
        type=str,
        required=True,
        default="nlp_summarization_train",
        help="Experiment name",
    )
    parser.add_argument(
        "--wait",
        default=False,
        action="store_true",
        help="wait for the job to finish",
    )

    group = parser.add_argument_group("Training parameters")
    group.add_argument(
        "--limit_samples",
        type=int,
        default=1000,
    )
    group.add_argument(
        "--pretrained_model_name",
        type=str,
        default="t5-small",
    )
    group.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    group.add_argument(
        "--learning_rate",
        type=float,
        default=0.00005,
    )
    group.add_argument(
        "--model_registration_name",
        type=str,
        default="pubmed-summarization",
    )

    group = parser.add_argument_group("Compute parameters")
    group.add_argument(
        "--cpu_compute",
        type=str,
        default="cpu-cluster",
    )
    group.add_argument(
        "--cpu_compute_large",
        type=str,
        default="cpu-cluster-lg",
    )
    group.add_argument(
        "--gpu_compute",
        type=str,
        default="gpu-cluster",
    )
    group.add_argument(
        "--training_nodes",
        type=int,
        default=1,
    )
    group.add_argument(
        "--gpus_per_node",
        type=int,
        default=1,
    )

    return parser


def connect_to_aml(args):
    """Connect to Azure ML workspace using provided cli arguments."""
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    # Get a handle to workspace
    try:
        # ml_client to connect using local config.json
        ml_client = MLClient.from_config(credential, path='config.json')

    except Exception as ex:
        print(
            "Could not find config.json, using config.yaml refs to Azure ML workspace instead."
        )

        # tries to connect using cli args if provided else using config.yaml
        ml_client = MLClient(
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name,
            credential=credential,
        )
    return ml_client


def build_components(args):
    """Builds the components for the pipeline."""
    DATA_SCIENCE_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..","..", "..", "data-science", "src"
    )

    prep_finetuning_dataset = command(
        name="prep_finetuning_dataset",
        display_name="Prepare dataset for training",
        inputs={
            "dataset_name": Input(type="string"),
            "dataset_config": Input(type="string"),
            "text_column": Input(type="string"),
            "summary_column": Input(type="string"),
            "limit_samples": Input(type="integer"),
            "max_input_length": Input(type="integer"),
            "max_target_length": Input(type="integer"),
            "padding": Input(type="string"),
            "pretrained_model_name": Input(type="string"),
        },
        outputs=dict(
            encodings=Output(type="uri_folder", mode="rw_mount"),
        ),
        code=DATA_SCIENCE_FOLDER,
        command="""python summarization/prepare.py \
                        --dataset_name ${{inputs.dataset_name}} \
                        --dataset_config ${{inputs.dataset_config}} \
                        --text_column ${{inputs.text_column}} \
                        --summary_column ${{inputs.summary_column}} \
                        --limit_samples ${{inputs.limit_samples}} \
                        --model_arch ${{inputs.pretrained_model_name}} \
                        --max_input_length ${{inputs.max_input_length}} \
                        --max_target_length ${{inputs.max_target_length}} \
                        --padding ${{inputs.padding}} \
                        --encodings ${{outputs.encodings}}\
                """,
        environment="nlp_summarization_train@latest",
    )

    finetune_model = command(
        name="finetune_model",
        display_name="Fine-tune summarization model",
        inputs={
            "preprocessed_datasets": Input(type="uri_folder"),
            "pretrained_model_name": Input(type="string"),
            "limit_samples": Input(type="integer"),
            "learning_rate": Input(type="number"),
            "num_train_epochs": Input(type="integer"),
            "per_device_train_batch_size": Input(type="integer"),
            "per_device_eval_batch_size": Input(type="integer"),
        },
        outputs=dict(
            finetuned_model=Output(type="uri_folder", mode="rw_mount"),
        ),
        code=DATA_SCIENCE_FOLDER,
        command="""python summarization/run.py \
                    --preprocessed_datasets ${{inputs.preprocessed_datasets}} \
                    --learning_rate ${{inputs.learning_rate}} \
                    --per_device_train_batch_size ${{inputs.per_device_train_batch_size}} \
                    --per_device_eval_batch_size ${{inputs.per_device_eval_batch_size}} \
                    --limit_samples ${{inputs.limit_samples}} \
                    --model_name ${{inputs.pretrained_model_name}} \
                    --model_output ${{outputs.finetuned_model}}\
                    --output_dir outputs \
                    --num_train_epochs ${{inputs.num_train_epochs}} \
                    --do_train --do_eval \
                """,
        environment="nlp_summarization_train@latest",
        distribution={
            "type": "PyTorch",
            # set process count to the number of gpus on the node
            "process_count_per_instance": args.gpus_per_node,
        },
        # set instance count to the number of nodes you want to use
        instance_count=args.training_nodes,
    )

    evaluate_model = command(
        name="evaluate_model",
        display_name="Run eval on a model",
        inputs={
            "preprocessed_datasets": Input(type="uri_folder"),
            "model_path": Input(type="uri_folder", optional=True),
            "model_name": Input(type="string", optional=True),
            "limit_samples": Input(type="integer"),
            "max_target_length": Input(type="integer"),
        },
        outputs=dict(
            metrics=Output(type="uri_folder", mode="rw_mount"),
        ),
        code=DATA_SCIENCE_FOLDER,
        command="""python summarization/run.py \
                    --preprocessed_datasets ${{inputs.preprocessed_datasets}} \
                    --limit_samples ${{inputs.limit_samples}} \
                    --output_dir ${{outputs.metrics}} \
                    $[[--model_path ${{inputs.model_path}}]] \
                    $[[--model_name ${{inputs.model_name}}]] \
                    --max_target_length ${{inputs.max_target_length}} \
                    --do_predict \
                """,
        environment="nlp_summarization_train@latest",
    )

    compare_models = command(
        name="compare_models",
        display_name="Compare finetuned to baseline",
        inputs={
            "baseline_metrics": Input(type="uri_folder"),
            "candidate_metrics": Input(type="uri_folder"),
            "reference_metric": Input(type="string"),
        },
        outputs=dict(
            deploy_flag=Output(type="uri_folder", mode="rw_mount"),
        ),
        code=DATA_SCIENCE_FOLDER,
        command="""python summarization/compare.py \
                    --baseline_metrics ${{inputs.baseline_metrics}} \
                    --candidate_metrics ${{inputs.candidate_metrics}} \
                    --reference_metric ${{inputs.reference_metric}} \
                    --deploy_flag ${{outputs.deploy_flag}} \
                    --force_comparison True\
                """,
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    )

    register_model = command(
        name="register_model",
        display_name="Register model",
        inputs={
            "model": Input(type="uri_folder"),
            "deploy_flag": Input(type="uri_folder"),
            "model_registration_name": Input(type="string"),
        },
        code=DATA_SCIENCE_FOLDER,
        command="""python summarization/register.py \
                    --model_folder ${{inputs.model}} \
                    --deploy_flag ${{inputs.deploy_flag}} \
                    --register_as ${{inputs.model_registration_name}} \
                """,
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    )

    return {
        "prep_finetuning_dataset": prep_finetuning_dataset,
        "finetune_model": finetune_model,
        "evaluate_model": evaluate_model,
        "compare_models": compare_models,
        "register_model": register_model,
    }


def main():
    """Main entry point for the script."""
    parser = get_config_parger()
    args, _ = parser.parse_known_args()
    ml_client = connect_to_aml(args)

    # get components from build function
    components_dict = build_components(args)
    prep_finetuning_dataset = components_dict["prep_finetuning_dataset"]
    finetune_model = components_dict["finetune_model"]
    evaluate_model = components_dict["evaluate_model"]
    compare_models = components_dict["compare_models"]
    register_model = components_dict["register_model"]

    # build the pipeline using Azure ML SDK v2
    @dsl.pipeline(
        name="NLP Training Pipeline",
        description="NLP Training Pipeline",
    )
    def nlp_training_pipeline(
        limit_samples: int,
        pretrained_model_name: str,
        num_train_epochs: int,
        batch_size: int,
        learning_rate: float,
        model_registration_name: str,
    ):
        prep_finetuning_dataset_step = prep_finetuning_dataset(
            dataset_name="ccdv/pubmed-summarization",
            dataset_config="section",
            text_column="article",
            summary_column="abstract",
            limit_samples=limit_samples,
            max_input_length=512,
            max_target_length=40,
            padding="max_length",
            pretrained_model_name=pretrained_model_name,
        )
        prep_finetuning_dataset_step.compute = args.cpu_compute_large

        finetune_model_step = finetune_model(
            preprocessed_datasets=prep_finetuning_dataset_step.outputs.encodings,
            pretrained_model_name=pretrained_model_name,
            limit_samples=limit_samples,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )
        finetune_model_step.compute = args.gpu_compute

        evaluate_finetuned_model_step = evaluate_model(
            preprocessed_datasets=prep_finetuning_dataset_step.outputs.encodings,
            model_path=finetune_model_step.outputs.finetuned_model,
            limit_samples=limit_samples,
            max_target_length=40,
        )
        evaluate_finetuned_model_step.compute = args.gpu_compute

        evaluate_baseline_model_step = evaluate_model(
            preprocessed_datasets=prep_finetuning_dataset_step.outputs.encodings,
            model_name=pretrained_model_name,
            limit_samples=limit_samples,
            max_target_length=40,
        )
        evaluate_baseline_model_step.compute = args.gpu_compute

        compare_models_step = compare_models(
            baseline_metrics=evaluate_finetuned_model_step.outputs.metrics,
            candidate_metrics=evaluate_baseline_model_step.outputs.metrics,
            reference_metric="predict_rougeLsum",
        )
        compare_models_step.compute = args.cpu_compute

        register_model_step = register_model(
            model=finetune_model_step.outputs.finetuned_model,
            deploy_flag=compare_models_step.outputs.deploy_flag,
            model_registration_name=model_registration_name,
        )
        register_model_step.compute = args.cpu_compute

    # instanciates the job
    pipeline_job = nlp_training_pipeline(
        limit_samples=args.limit_samples,
        pretrained_model_name=args.pretrained_model_name,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_registration_name=args.model_registration_name,
    )

    # submits the job
    print("Submitting the pipeline job to your AzureML workspace...")
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.n
    )

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)

    if args.wait:
        ml_client.jobs.stream(pipeline_job.name)


if __name__ == "__main__":
    main()
