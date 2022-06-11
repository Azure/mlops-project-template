# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Original Author: Jeff Omhover (MSFT)


"""
This script implements a Distributed PyTorch training sequence.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed pytorch
- MLFLOW : how to implement mlflow reporting of metrics and artifacts
- PROFILER : how to implement pytorch profiler
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from distutils.util import strtobool

import mlflow

# the long list of torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.utils import ModelOutput

# add path to here, if necessary
COMPONENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if COMPONENT_ROOT not in sys.path:
    logging.info(f"Adding {COMPONENT_ROOT} to path")
    sys.path.append(str(COMPONENT_ROOT))

from image_io import build_image_datasets

# internal imports
from model import get_model_metadata, load_model
from profiling import (
    LogDiskIOBlock,
    LogTimeBlock,
    LogTimeOfIterator,
    PyTorchProfilerHandler,
)

torch.set_default_dtype(torch.float64)


class PyTorchDistributedModelTrainingSequence:
    """Generic class to run the sequence for training a PyTorch model
    using distributed training."""

    def __init__(self):
        """Constructor"""
        self.logger = logging.getLogger(__name__)

        # DATA
        self.training_data_sampler = None
        self.training_data_loader = None
        self.validation_data_loader = None

        # MODEL
        self.model = None
        self.labels = []
        self.model_signature = None

        # DISTRIBUTED CONFIG
        self.world_size = 1
        self.world_rank = 0
        self.local_world_size = 1
        self.local_rank = 0
        self.multinode_available = False
        self.cpu_count = os.cpu_count()
        self.device = None
        # NOTE: if we're running multiple nodes, this indicates if we're on first node
        self.self_is_main_node = True

        # TRAINING CONFIGS
        self.dataloading_config = None
        self.training_config = None

        # PROFILER
        self.profiler = None
        self.profiler_output_tmp_dir = None

    #####################
    ### SETUP METHODS ###
    #####################

    def setup_config(self, args):
        """Sets internal variables using provided CLI arguments (see build_arguments_parser()).
        In particular, sets device(cuda) and multinode parameters."""
        self.dataloading_config = args
        self.training_config = args

        # verify parameter default values
        if self.dataloading_config.num_workers is None:
            self.dataloading_config.num_workers = 0
        if self.dataloading_config.num_workers < 0:
            self.dataloading_config.num_workers = self.cpu_count
        if self.dataloading_config.num_workers == 0:
            self.logger.warning(
                "You specified num_workers=0, forcing prefetch_factor to be discarded."
            )
            self.dataloading_config.prefetch_factor = None

        # NOTE: strtobool returns an int, converting to bool explicitely
        self.dataloading_config.pin_memory = bool(self.dataloading_config.pin_memory)
        self.dataloading_config.non_blocking = bool(
            self.dataloading_config.non_blocking
        )

        # add this switch to test for different strategies
        if self.dataloading_config.multiprocessing_sharing_strategy:
            torch.multiprocessing.set_sharing_strategy(
                self.dataloading_config.multiprocessing_sharing_strategy
            )

        # DISTRIBUTED: detect multinode config
        # depending on the Azure ML distribution.type, different environment variables will be provided
        # to configure DistributedDataParallel
        self.distributed_backend = args.distributed_backend
        if self.distributed_backend == "nccl":
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.world_rank = int(os.environ.get("RANK", "0"))
            self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.multinode_available = self.world_size > 1
            self.self_is_main_node = self.world_rank == 0

        elif self.distributed_backend == "mpi":
            # Note: Distributed pytorch package doesn't have MPI built in.
            # MPI is only included if you build PyTorch from source on a host that has MPI installed.
            self.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
            self.world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
            self.local_world_size = int(
                os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")
            )
            self.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
            self.multinode_available = self.world_size > 1
            self.self_is_main_node = self.world_rank == 0

        else:
            raise NotImplementedError(
                f"distributed_backend={self.distributed_backend} is not implemented yet."
            )

        # Use CUDA if it is available
        if not self.training_config.disable_cuda and torch.cuda.is_available():
            self.logger.info(
                f"Setting up torch.device for CUDA for local gpu:{self.local_rank}"
            )
            self.device = torch.device(self.local_rank)
        else:
            self.logger.info(f"Setting up torch.device for cpu")
            self.device = torch.device("cpu")

        if self.multinode_available:
            self.logger.info(
                f"Running in multinode with backend={self.distributed_backend} local_rank={self.local_rank} rank={self.world_rank} size={self.world_size}"
            )
            # DISTRIBUTED: this is required to initialize the pytorch backend
            torch.distributed.init_process_group(
                self.distributed_backend,
                rank=self.world_rank,
                world_size=self.world_size,
            )
        else:
            self.logger.info(
                f"Not running in multinode, so not initializing process group."
            )

        # DISTRIBUTED: in distributed mode, you want to report parameters
        # only from main process (rank==0) to avoid conflict
        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            logged_params = {
                # log some distribution params
                "nodes": int(os.environ.get("AZUREML_NODE_COUNT", "1")),
                "instance_per_node": self.world_size
                // int(os.environ.get("AZUREML_NODE_COUNT", "1")),
                "cuda_available": torch.cuda.is_available(),
                "disable_cuda": self.training_config.disable_cuda,
                "distributed": self.multinode_available,
                "distributed_backend": self.distributed_backend,
                # data loading params
                "batch_size": self.dataloading_config.batch_size,
                "num_workers": self.dataloading_config.num_workers,
                "cpu_count": self.cpu_count,
                "prefetch_factor": self.dataloading_config.prefetch_factor,
                "persistent_workers": self.dataloading_config.persistent_workers,
                "pin_memory": self.dataloading_config.pin_memory,
                "non_blocking": self.dataloading_config.non_blocking,
                "multiprocessing_sharing_strategy": self.dataloading_config.multiprocessing_sharing_strategy,
                # training params
                "model_arch": self.training_config.model_arch,
                "model_arch_pretrained": self.training_config.model_arch_pretrained,
                "optimizer.learning_rate": self.training_config.learning_rate,
                "optimizer.momentum": self.training_config.momentum,
                # profiling params
                "enable_profiling": self.training_config.enable_profiling,
            }

            if not self.training_config.disable_cuda and torch.cuda.is_available():
                # add some gpu properties
                logged_params["cuda_device_count"] = torch.cuda.device_count()
                cuda_device_properties = torch.cuda.get_device_properties(self.device)
                logged_params["cuda_device_name"] = cuda_device_properties.name
                logged_params["cuda_device_major"] = cuda_device_properties.major
                logged_params["cuda_device_minor"] = cuda_device_properties.minor
                logged_params[
                    "cuda_device_memory"
                ] = cuda_device_properties.total_memory
                logged_params[
                    "cuda_device_processor_count"
                ] = cuda_device_properties.multi_processor_count

            mlflow.log_params(logged_params)

    def setup_datasets(
        self,
        training_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        labels: list,
    ):
        """Creates and sets up dataloaders for training/validation datasets."""
        self.labels = labels

        # DISTRIBUTED: you need to use a DistributedSampler that wraps your dataset
        # it will draw a different sample on each node/process to distribute data sampling
        self.training_data_sampler = DistributedSampler(
            training_dataset, num_replicas=self.world_size, rank=self.world_rank
        )

        # setting up DataLoader with the right arguments
        optional_data_loading_kwargs = {}

        if self.dataloading_config.num_workers > 0:
            # NOTE: this option _ONLY_ applies if num_workers > 0
            # or else DataLoader will except
            optional_data_loading_kwargs[
                "prefetch_factor"
            ] = self.dataloading_config.prefetch_factor
            optional_data_loading_kwargs[
                "persistent_workers"
            ] = self.dataloading_config.persistent_workers

        self.training_data_loader = DataLoader(
            training_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            pin_memory=self.dataloading_config.pin_memory,
            # DISTRIBUTED: the sampler needs to be provided to the DataLoader
            sampler=self.training_data_sampler,
            # all other args
            **optional_data_loading_kwargs,
        )

        # DISTRIBUTED: we don't need a sampler for validation set
        # it is used as-is in every node/process
        self.validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=self.dataloading_config.batch_size,
            num_workers=self.dataloading_config.num_workers,  # self.cpu_count,
            pin_memory=self.dataloading_config.pin_memory,
        )

        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            mlflow.log_params({"num_classes": len(labels)})

    def setup_model(self, model):
        """Configures a model for training."""
        self.logger.info(f"Setting up model to use device {self.device}")
        self.model = model.to(self.device)

        # DISTRIBUTED: the model needs to be wrapped in a DistributedDataParallel class
        if self.multinode_available:
            self.logger.info(f"Setting up model to use DistributedDataParallel.")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        # fun: log the number of parameters
        params_count = 0
        for param in model.parameters():
            if param.requires_grad:
                params_count += param.numel()
        self.logger.info(
            "MLFLOW: model_param_count={:.2f} (millions)".format(
                round(params_count / 1e6, 2)
            )
        )
        if self.self_is_main_node:
            mlflow.log_params({"model_param_count": round(params_count / 1e6, 2)})

        return self.model

    ########################
    ### TRAINING METHODS ###
    ########################

    def _epoch_eval(self, epoch, criterion):
        """Called during train() for running the eval phase of one epoch."""
        with torch.no_grad():
            num_correct = 0
            num_total_images = 0
            running_loss = 0.0

            epoch_eval_metrics = {}

            # PROFILER: here we're introducing a layer on top of data loader to capture its performance
            # in pratice, we'd just use for images, targets in tqdm(self.training_data_loader)
            for images, targets in LogTimeOfIterator(
                tqdm(self.validation_data_loader),
                "validation_data_loader",
                async_collector=epoch_eval_metrics,
            ):
                with record_function("eval.to_device"):
                    images = images.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )
                    targets = targets.to(
                        self.device, non_blocking=self.dataloading_config.non_blocking
                    )

                with record_function("eval.forward"):
                    outputs = self.model(images)

                    if isinstance(outputs, torch.Tensor):
                        # if we're training a regular pytorch model (ex: torchvision)
                        loss = criterion(outputs, targets)
                        _, predicted = torch.max(outputs.data, 1)
                        correct = predicted == targets
                    elif isinstance(outputs, ModelOutput):
                        # if we're training a HuggingFace model
                        loss = criterion(outputs.logits, targets)
                        _, predicted = torch.max(outputs.logits.data, 1)
                        correct = predicted == targets
                    else:
                        # if anything else, just except
                        raise ValueError(
                            f"outputs from model is type {type(outputs)} which is unknown."
                        )

                    running_loss += loss.item() * images.size(0)

                    num_correct += torch.sum(correct).item()
                    num_total_images += len(images)

        epoch_eval_metrics["running_loss"] = running_loss
        epoch_eval_metrics["num_correct"] = num_correct
        epoch_eval_metrics["num_samples"] = num_total_images

        return epoch_eval_metrics

    def _epoch_train(self, epoch, optimizer, scheduler, criterion):
        """Called during train() for running the train phase of one epoch."""
        self.model.train()
        self.training_data_sampler.set_epoch(epoch)

        num_correct = 0
        num_total_images = 0
        running_loss = 0.0

        epoch_train_metrics = {}

        # PROFILER: here we're introducing a layer on top of data loader to capture its performance
        # in pratice, we'd just use for images, targets in tqdm(self.training_data_loader)
        for images, targets in LogTimeOfIterator(
            tqdm(self.training_data_loader),
            "training_data_loader",
            async_collector=epoch_train_metrics,
        ):
            # PROFILER: record_function will report to the profiler (if enabled)
            # here a specific wall time for a given block of code
            with record_function("train.to_device"):
                images = images.to(
                    self.device, non_blocking=self.dataloading_config.non_blocking
                )
                targets = targets.to(
                    self.device, non_blocking=self.dataloading_config.non_blocking
                )

            with record_function("train.forward"):
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.model(images)

                # if self.model_signature is None:
                #     self.model_signature = infer_signature(images, outputs)

                if isinstance(outputs, torch.Tensor):
                    # if we're training a regular pytorch model (ex: torchvision)
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = predicted == targets
                elif isinstance(outputs, ModelOutput):
                    # if we're training a HuggingFace model
                    loss = criterion(outputs.logits, targets)
                    _, predicted = torch.max(outputs.logits.data, 1)
                    correct = predicted == targets
                else:
                    # if anything else, just except
                    raise ValueError(
                        f"outputs from model is type {type(outputs)} which is unknown."
                    )

                running_loss += loss.item() * images.size(0)
                num_correct += torch.sum(correct).item()
                num_total_images += len(images)

            # PROFILER: record_function will report to the profiler (if enabled)
            # here a specific wall time for a given block of code
            with record_function("train.backward"):
                loss.backward()
                optimizer.step()
                scheduler.step()

        epoch_train_metrics["running_loss"] = running_loss
        epoch_train_metrics["num_correct"] = num_correct
        epoch_train_metrics["num_samples"] = num_total_images

        return epoch_train_metrics

    def train(self, epochs: int = None, checkpoints_dir: str = None):
        """Trains the model.

        Args:
            epochs (int, optional): if not provided uses internal config
            checkpoints_dir (str, optional): path to write checkpoints
        """
        if epochs is None:
            epochs = self.training_config.num_epochs

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            momentum=self.training_config.momentum,
            nesterov=True,
            # weight_decay=1e-4,
        )

        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # DISTRIBUTED: export checkpoint only from main node
        if self.self_is_main_node and checkpoints_dir is not None:
            # saving checkpoint before training
            self.checkpoint_save(
                self.model, optimizer, checkpoints_dir, epoch=-1, loss=0.0
            )

        # DISTRIBUTED: you'll node that this loop has nothing specifically "distributed"
        # that's because most of the changes are in the backend (DistributedDataParallel)
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch={epoch}")

            # we'll collect metrics we want to report for this epoch
            epoch_metrics = {}

            # start timer for epoch time metric
            epoch_train_start = time.time()

            # TRAIN: loop on training set and return metrics
            epoch_train_metrics = self._epoch_train(
                epoch, optimizer, scheduler, criterion
            )
            self.logger.info(f"Epoch metrics: {epoch_train_metrics}")

            # stop timer
            epoch_metrics["epoch_train_time"] = time.time() - epoch_train_start

            # record metrics of interest
            epoch_metrics["training_data_loader.count"] = epoch_train_metrics[
                "training_data_loader.count"
            ]
            epoch_metrics["training_data_loader.time.sum"] = epoch_train_metrics[
                "training_data_loader.time.sum"
            ]
            epoch_metrics["training_data_loader.time.first"] = epoch_train_metrics[
                "training_data_loader.time.first"
            ]
            epoch_metrics["epoch_train_loss"] = (
                epoch_train_metrics["running_loss"] / epoch_train_metrics["num_samples"]
            )
            epoch_metrics["epoch_train_acc"] = (
                epoch_train_metrics["num_correct"] / epoch_train_metrics["num_samples"]
            )

            # start timer for epoch time metric
            epoch_eval_start = time.time()

            # EVAL: run evaluation on validation set and return metrics
            epoch_eval_metrics = self._epoch_eval(epoch, criterion)
            self.logger.info(f"Epoch metrics: {epoch_train_metrics}")

            # stop timer
            epoch_metrics["epoch_eval_time"] = time.time() - epoch_eval_start

            # record metrics of interest
            epoch_metrics["validation_data_loader.count"] = epoch_eval_metrics[
                "validation_data_loader.count"
            ]
            epoch_metrics["validation_data_loader.time.sum"] = epoch_eval_metrics[
                "validation_data_loader.time.sum"
            ]
            epoch_metrics["validation_data_loader.time.first"] = epoch_eval_metrics[
                "validation_data_loader.time.first"
            ]
            epoch_metrics["epoch_valid_loss"] = (
                epoch_eval_metrics["running_loss"] / epoch_eval_metrics["num_samples"]
            )
            epoch_metrics["epoch_valid_acc"] = (
                epoch_eval_metrics["num_correct"] / epoch_eval_metrics["num_samples"]
            )

            # start timer for epoch time metric
            epoch_utility_start = time.time()

            # PROFILER: use profiler.step() to mark a step in training
            # the pytorch profiler will use internally to trigger
            # saving the traces in different files
            if self.profiler:
                self.profiler.step()

            # DISTRIBUTED: export checkpoint only from main node
            if self.self_is_main_node and checkpoints_dir is not None:
                self.checkpoint_save(
                    self.model,
                    optimizer,
                    checkpoints_dir,
                    epoch=epoch,
                    loss=epoch_metrics["epoch_valid_loss"],
                )

            # report metric values in stdout
            self.logger.info(f"MLFLOW: metrics={epoch_metrics} epoch={epoch}")

            # MLFLOW / DISTRIBUTED: report metrics only from main node
            if self.self_is_main_node:
                mlflow.log_metrics(epoch_metrics)
                mlflow.log_metric(
                    "epoch_utility_time", time.time() - epoch_utility_start, step=epoch
                )

    def runtime_error_report(self, runtime_exception):
        """Call this when catching a critical exception.
        Will print all sorts of relevant information to the log."""
        self.logger.critical(traceback.format_exc())
        try:
            import psutil

            self.logger.critical(f"Memory: {str(psutil.virtual_memory())}")
        except ModuleNotFoundError:
            self.logger.critical(
                "import psutil failed, cannot display virtual memory stats."
            )

        if torch.cuda.is_available():
            self.logger.critical(
                "Cuda memory summary:\n"
                + str(torch.cuda.memory_summary(device=None, abbreviated=False))
            )
            self.logger.critical(
                "Cuda memory snapshot:\n"
                + json.dumps(torch.cuda.memory_snapshot(), indent="    ")
            )
        else:
            self.logger.critical(
                "Cuda is not available, cannot report cuda memory allocation."
            )

    def close(self):
        """Tear down potential resources"""
        if self.multinode_available:
            self.logger.info(
                f"Destroying process group on local_rank={self.local_rank} rank={self.world_rank} size={self.world_size}"
            )
            # DISTRIBUTED: this will teardown the distributed process group
            torch.distributed.destroy_process_group()
        else:
            self.logger.info(
                f"Not running in multinode, so not destroying process group."
            )

    #################
    ### MODEL I/O ###
    #################

    def checkpoint_save(
        self, model, optimizer, output_dir: str, epoch: int, loss: float
    ):
        """Saves model as checkpoint"""
        # create output directory just in case
        os.makedirs(output_dir, exist_ok=True)

        model_output_path = os.path.join(
            output_dir, f"model-checkpoint-epoch{epoch}-loss{loss}.pt"
        )

        self.logger.info(f"Exporting checkpoint to {model_output_path}")

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # DISTRIBUTED: to export model, you need to get it out of the DistributedDataParallel class
            self.logger.info(
                "Model was distributed, we will checkpoint DistributedDataParallel.module"
            )
            model_to_save = model.module
        else:
            model_to_save = model

        with record_function("checkpoint.save"):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                model_output_path,
            )

    def save(
        self,
        output_dir: str,
        name: str = "dev",
        register_as: str = None,
    ) -> None:
        # DISTRIBUTED: you want to save the model only from the main node/process
        # in data distributed mode, all models should theoretically be the same
        if self.self_is_main_node:
            self.logger.info(f"Saving model and classes in {output_dir}...")

            # create output directory just in case
            os.makedirs(output_dir, exist_ok=True)

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                # DISTRIBUTED: to export model, you need to get it out of the DistributedDataParallel class
                self.logger.info(
                    "Model was distributed, we will export DistributedDataParallel.module"
                )
                model_to_save = self.model.module.to("cpu")
            else:
                model_to_save = self.model.to("cpu")

            # Save the labels to a csv file.
            # This file will be required to map the output array
            # from the API to the labels.
            with open("label-mapping.txt", "w") as f:
                f.write("\n".join(self.labels))
            mlflow.log_artifact("label-mapping.txt")

            # MLFLOW: mlflow has a nice method to export the model automatically
            # add tags and environment for it. You can then use it in Azure ML
            # to register your model to an endpoint.
            mlflow.pytorch.log_model(
                model_to_save,
                artifact_path="final_model",
                registered_model_name=register_as,  # also register it if name is provided
                signature=self.model_signature,
            )

            # MLFLOW: Register the model with the model registry
            # This is useful for Azure ML to register your model
            # to an endpoint.
            if register_as is not None:
                mlflow.register_model(
                    model_uri=f"runs:/{mlflow.active_run().info.run_id}/final_model",
                    name=register_as,
                )


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group(f"Training Inputs")
    group.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="Path to folder containing training images",
    )
    group.add_argument(
        "--valid_images",
        type=str,
        required=True,
        help="path to folder containing validation images",
    )

    group = parser.add_argument_group(f"Training Outputs")
    group.add_argument(
        "--model_output",
        type=str,
        required=False,
        default=None,
        help="Path to write final model",
    )
    group.add_argument(
        "--checkpoints",
        type=str,
        required=False,
        default=None,
        help="Path to read/write checkpoints",
    )
    group.add_argument(
        "--register_model_as",
        type=str,
        required=False,
        default=None,
        help="Name to register final model in MLFlow",
    )

    group = parser.add_argument_group(f"Data Loading Parameters")
    group.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Train/valid data loading batch size (default: 64)",
    )
    group.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=None,
        help="Num workers for data loader (default: -1 => all cpus available)",
    )
    group.add_argument(
        "--prefetch_factor",
        type=int,
        required=False,
        default=2,
        help="Data loader prefetch factor (default: 2)",
    )
    group.add_argument(
        "--persistent_workers",
        type=strtobool,
        required=False,
        default=True,
        help="Use persistent prefetching workers (default: True)",
    )
    group.add_argument(
        "--pin_memory",
        type=strtobool,
        required=False,
        default=True,
        help="Pin Data loader prefetch factor (default: True)",
    )
    group.add_argument(
        "--non_blocking",
        type=strtobool,
        required=False,
        default=False,
        help="Use non-blocking transfer to device (default: False)",
    )

    group = parser.add_argument_group(f"Model/Training Parameters")
    group.add_argument(
        "--model_arch",
        type=str,
        required=False,
        default="resnet18",
        help="Which model architecture to use (default: resnet18)",
    )
    group.add_argument(
        "--model_arch_pretrained",
        type=strtobool,
        required=False,
        default=True,
        help="Use pretrained model (default: true)",
    )
    group.add_argument(
        "--distributed_backend",
        type=str,
        required=False,
        choices=["nccl", "mpi"],
        default="nccl",
        help="Which distributed backend to use.",
    )
    group.add_argument(
        "--disable_cuda",
        type=strtobool,
        required=False,
        default=False,
        help="set True to force use of cpu (local testing).",
    )
    # DISTRIBUTED: torch.distributed.launch is passing this argument to your script
    # it is likely to be deprecated in favor of os.environ['LOCAL_RANK']
    # see https://pytorch.org/docs/stable/distributed.html#launch-utility
    group.add_argument(
        "--local_rank",
        type=int,
        required=False,
        default=None,
        help="Passed by torch.distributed.launch utility when running from cli.",
    )
    group.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=1,
        help="Number of epochs to train for",
    )
    group.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate of optimizer",
    )
    group.add_argument(
        "--momentum",
        type=float,
        required=False,
        default=0.9,
        help="Momentum of optimizer",
    )

    group = parser.add_argument_group(f"System Parameters")
    group.add_argument(
        "--enable_profiling",
        type=strtobool,
        required=False,
        default=False,
        help="Enable pytorch profiler.",
    )
    group.add_argument(
        "--multiprocessing_sharing_strategy",
        type=str,
        choices=torch.multiprocessing.get_all_sharing_strategies(),
        required=False,
        default=None,
        help="Check https://pytorch.org/docs/stable/multiprocessing.html",
    )

    return parser


def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # use a handler for the training sequence
    training_handler = PyTorchDistributedModelTrainingSequence()

    # sets cuda and distributed config
    training_handler.setup_config(args)

    # PROFILER: here we use a helper class to enable profiling
    # see profiling.py for the implementation details
    training_profiler = PyTorchProfilerHandler(
        enabled=bool(args.enable_profiling),
        rank=training_handler.world_rank,
    )
    # PROFILER: set profiler in trainer to call profiler.step() during training
    training_handler.profiler = training_profiler.start_profiler()

    # report the time and disk usage during this code block
    with LogTimeBlock(
        "build_image_datasets", enabled=training_handler.self_is_main_node
    ), LogDiskIOBlock(
        "build_image_datasets", enabled=training_handler.self_is_main_node
    ):
        # build the image folder datasets
        train_dataset, valid_dataset, labels = build_image_datasets(
            train_images_dir=args.train_images,
            valid_images_dir=args.valid_images,
            input_size=get_model_metadata(args.model_arch)["input_size"],
        )

    # creates data loaders from datasets for distributed training
    training_handler.setup_datasets(train_dataset, valid_dataset, labels)

    with LogTimeBlock("load_model", enabled=training_handler.self_is_main_node):
        # creates the model architecture
        model = load_model(
            args.model_arch,
            output_dimension=len(labels),
            pretrained=args.model_arch_pretrained,
        )

    # logging of labels
    logger.info(labels)
    # sets the model for distributed training
    training_handler.setup_model(model)

    # runs training sequence
    # NOTE: num_epochs is provided in args
    try:
        training_handler.train(checkpoints_dir=args.checkpoints)
    except RuntimeError as runtime_exception:  # if runtime error occurs (ex: cuda out of memory)
        # then print some runtime error report in the logs
        training_handler.runtime_error_report(runtime_exception)
        # re-raise
        raise runtime_exception

    # stops profiling (and save in mlflow)
    training_profiler.stop_profiler()

    # saves final model
    if args.model_output:
        training_handler.save(
            args.model_output,
            name=f"epoch-{args.num_epochs}",
            register_as=args.register_model_as,
        )

    # properly teardown distributed resources
    training_handler.close()

    # MLFLOW: finalize mlflow (once in entire script)
    mlflow.end_run()

    logger.info("run() completed")


def main(cli_args=None):
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

    # create argument parser
    parser = build_arguments_parser()

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # run the run function
    run(args)


if __name__ == "__main__":
    main()
