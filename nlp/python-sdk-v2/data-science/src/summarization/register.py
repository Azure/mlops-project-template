from azureml.core import Run
from azureml.core.model import Model

import os
import argparse
import logging
import mlflow


def main():
    """Main function of the script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="folder containing model",
    )
    parser.add_argument(
        "--register_as",
        type=str,
        required=True,
        help="name to use for model registration in AzureML",
    )
    parser.add_argument(
        "--deploy_flag", type=str, required=True, help="a deploy flag whether to deploy or not"
    )

    args = parser.parse_args()
    logger.info(f"Running with arguments: {args}")

    # Start Logging
    mlflow.start_run()

    if os.path.isfile(args.deploy_flag):
        deploy_flag_file_path = args.deploy_flag
    else:
        deploy_flag_file_path = os.path.join(args.deploy_flag, "deploy_flag")

    logger.info(f"Opening deploy_flag file from {deploy_flag_file_path}")
    with open(deploy_flag_file_path, 'rb') as in_file:
        deploy_flag = bool(int(in_file.read()))

    if deploy_flag:
        logger.info(f"Deploy flag is True, registering model as {args.register_as}...")
        run = Run.get_context()

        # if we're running locally, except
        if run.__class__.__name__ == "_OfflineRun":
            raise Exception("You can't run this script locally, you will need to run it as an AzureML job.")

        _ = Model.register(
            run.experiment.workspace,
            model_name=args.register_as,
            model_path=args.model_folder,
            tags={
                "type": "huggingface",
                "task": "summarization"
            },
            description="Huggingface model finetuned for summarization",
        )
    else:
        logger.info(f"Deploy flag is False, pass.")

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()

