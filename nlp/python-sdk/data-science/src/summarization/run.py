import logging
import os
from datasets import load_metric, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
    IntervalStrategy,
)
from transformers.trainer_callback import TrainerCallback

import torch
import nltk
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import mlflow
from pynvml import *
import time


# Input arguments are set with dataclass. Huggingface library stores the default training args in TrainingArguments dataclass
# user args are also defined in dataclasses, we will then load arguments from a tuple of user defined and built-in dataclasses.
@dataclass
class DataArgs:
    # Inputs
    preprocessed_datasets: str = field(
        default=None, metadata={"help": "path to preprocesed datasets"}
    )

    # Processing parameters
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "maxi sequence length for target text after tokenization."},
    )
    limit_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "limit the number of samples for faster run."},
    )


@dataclass
class ModelArgs:
    model_name: Optional[str] = field(default=None, metadata={"help": "model name"})
    model_path: Optional[str] = field(
        default=None, metadata={"help": "path to existing model file to load"}
    )
    model_output: Optional[str] = field(
        default=None, metadata={"help": "path to save the model"}
    )


nltk.download("punkt")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def postprocess_text(preds, labels):
    """Postprocess output for computing metrics"""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, metric):
    """Compute metric based on predictions from evaluation"""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


class CustomCallback(TrainerCallback):
    """A [`TrainerCallback`] that sends the logs to [AzureML](https://pypi.org/project/azureml-sdk/).

    This is a hotfix for the issue raised here:
    https://github.com/huggingface/transformers/issues/18870
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
            mlflow.log_metrics(metrics=metrics, step=state.global_step)


def main():
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # initialize the mlflow session
    mlflow.start_run()

    parser = HfArgumentParser((ModelArgs, DataArgs, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(f"Running with arguments: {model_args}, {data_args}, {training_args}")

    # Check if this is the main node
    is_this_main_node = int(os.environ.get("RANK", "0")) == 0
    if is_this_main_node:
        logger.info("This is the main Node")

    input_datasets = load_from_disk(data_args.preprocessed_datasets)
    logger.info(f"preprocessed dataset is loaded")

    if model_args.model_path:
        logger.info("using a saved model")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    else:
        logger.info("using a model from model library")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    # Artificially limit the number of samples (for testing)
    if training_args.do_train:  # if using --do-train from Seq2SeqTrainingArguments
        if data_args.limit_samples > 0:
            max_train_samples = min(len(input_datasets["train"]), data_args.limit_samples)
            train_dataset = input_datasets["train"].select(range(max_train_samples))
            logger.info(f"train: making a {max_train_samples} sample of the data")
        else:
            train_dataset = input_datasets["train"]

    if training_args.do_eval:
        if data_args.limit_samples > 0:
            max_eval_samples = min(
                len(input_datasets["validation"]), data_args.limit_samples
            )
            eval_dataset = input_datasets["validation"].select(range(max_eval_samples))
            logger.info(f"eval: making a {max_eval_samples} sample of the data")
        else:
            eval_dataset = input_datasets["validation"]

    if training_args.do_predict:
        if data_args.limit_samples > 0:
            max_predict_samples = min(
                len(input_datasets["test"]), data_args.limit_samples
            )
            predict_dataset = input_datasets["test"].select(range(max_predict_samples))
            logger.info(f"predict: making a {max_predict_samples} sample of the data")
        else:
            predict_dataset = input_datasets["test"]

    # Data collator
    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # Metric
    metric = load_metric("rouge")

    if training_args.do_train:
        logging_steps = len(train_dataset) // training_args.per_device_train_batch_size
        training_args.logging_steps = logging_steps
    #training_args.output_dir = "outputs"
    training_args.save_strategy = "epoch"
    training_args.evaluation_strategy = IntervalStrategy.EPOCH
    training_args.predict_with_generate = True
    training_args.report_to = [] # use our own callback
    logger.info(f"training args: {training_args}")

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer, metric),
        callbacks=[CustomCallback]
    )

    # Start the actual training (to include evaluation use --do-eval)
    if training_args.do_train:
        logger.info("Start training")
        start = time.time()
        train_result = trainer.train()

        mlflow.log_metric(
            "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
        )
        logger.info(
            "training is done"
        )  # Only print gpu utilization if gpu is available
        if torch.cuda.is_available():
            print_summary(train_result)

    # Save the model as an output
    if model_args.model_output and is_this_main_node:
        logger.info(f"Saving the model at {model_args.model_output}")
        os.makedirs(model_args.model_output, exist_ok=True)
        trainer.save_model(model_args.model_output)

    # Just run the predictions
    if training_args.do_predict:
        logger.info("*** Predict ***")
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.max_target_length
        )

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
