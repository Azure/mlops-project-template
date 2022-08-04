import os
import argparse
import logging
import mlflow
import json
from distutils.util import strtobool

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
        "--baseline_metrics",
        type=str,
        required=True,
        help="path to baseline metrics folder containing all_results.json",
    )
    parser.add_argument(
        "--candidate_metrics",
        type=str,
        required=True,
        help="path to candidate metrics folder containing all_results.json",
    )
    parser.add_argument(
        "--reference_metric",
        type=str,
        default="predict_rougeLsum",
        help="name of reference metric for shipping flag (default: predict_rougeLsum)",
    )
    parser.add_argument(
        "--force_comparison", type=strtobool, default=False, help="set to True to bypass comparison and set --deploy_flag to True"
    )
    parser.add_argument(
        "--deploy_flag", type=str, help="a deploy flag whether to deploy or not"
    )

    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    logger.info(f"Running with arguments: {args}")

    # open metrics on both sides
    with open(os.path.join(args.baseline_metrics, "all_results.json")) as in_file:
        baseline_metrics = json.loads(in_file.read())
    with open(os.path.join(args.candidate_metrics, "all_results.json")) as in_file:
        candidate_metrics = json.loads(in_file.read())

    # should we ship or not?
    if args.force_comparison:
        deploy_flag = True
    else:
        deploy_flag = (
            candidate_metrics[args.reference_metric]
            > baseline_metrics[args.reference_metric]
        )

    logger.info("baseline_metrics[{}]={}, candidate_metrics[{}]={}, deploy_flag={} (force_comparison={})".format(
        args.reference_metric,
        baseline_metrics[args.reference_metric],
        args.reference_metric,
        candidate_metrics[args.reference_metric],
        deploy_flag,
        args.force_comparison
    ))

    # save deploy_flag as a file
    os.makedirs(args.deploy_flag, exist_ok=True)
    with open(os.path.join(args.deploy_flag, "deploy_flag"), "w") as out_file:
        out_file.write("%d" % int(deploy_flag))

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
