# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Original Author: Jeff Omhover (MSFT)

"""
This script provides some helper code to help with pytorch profiling.
"""
import os
import time
import logging
import torch
import mlflow
import tempfile
from torch.profiler import ProfilerActivity
from typing import Any


def markdown_trace_handler(dir_name: str, rank: int = 0):
    """This handler can be used inside torch.profiler call to output
    tables in markdown format"""

    def _handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)

        # Note: trying to identify a unique name for the file
        file_name = os.path.join(
            dir_name,
            f"stacks_rank{rank}_step{prof.step_num}_t{int(time.time() * 1000)}.ms",
        )

        logging.getLogger(__name__).info(
            f"Exporting profiler trace as markdown at {file_name}"
        )
        # generate report in markdown format
        markdown = ["# Pytorch Profiler report"]

        markdown.append("## Average by cuda time")
        markdown.append("```")
        markdown.append(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        )
        markdown.append("```")

        with open(file_name, "w") as out_file:
            out_file.write("\n".join(markdown))

    return _handler_fn


def composite_trace_handler(handler_list):
    """This can call multiple trace handlers inside one"""

    def _handler_fn(prof) -> None:
        for handler in handler_list:
            handler(prof)

    return _handler_fn


def export_stack_trace_handler(
    dir_name: str, rank: int = 0, metrics=["self_cuda_time_total"]
):
    """This handler can be used inside torch.profiler call to output
    tables in markdown format"""

    def _handler_fn(prof) -> None:
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)

        # Note: trying to identify a unique name for the file
        for metric in metrics:
            file_name = os.path.join(
                dir_name,
                f"stacks_{metric}_rank{rank}_step{prof.step_num}_t{ int(time.time() * 1000)}.txt",
            )

            logging.getLogger(__name__).info(
                f"Exporting {metric} stacks as text at {file_name}"
            )

            prof.export_stacks(file_name, metric)

    return _handler_fn


class PyTorchProfilerHandler:
    """This class handles the initialization and setup of PyTorch profiler"""

    def __init__(self, enabled=False, rank=None):
        """Constructor.

        Args:
            enabled (bool): is profiling enabled?
            export_format (str): generate 'markdown' or 'tensorboard' profile in mlflow artifacts
            rank (int): rank of the current process/node
        """
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        self.rank = rank
        self.profiler_output_tmp_dir = None
        self.profiler = None

    def start_profiler(self):
        """Setup and start the pytorch profiler.

        Returns:
            profiler (torch.profiler): the profiler
        """
        if self.enabled:
            self.profiler_output_tmp_dir = tempfile.TemporaryDirectory()
            self.logger.info(
                f"Starting profiler (enabled=True) with tmp dir {self.profiler_output_tmp_dir.name}."
            )

            ## profiler activities CPU/GPU
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                self.logger.info(f"Enabling CUDA in profiler.")
                activities.append(ProfilerActivity.CUDA)

            ## handlers for exporting profile at each step
            # we're creating a list to export in multiple formats
            trace_handlers = []

            # export in markdown
            markdown_logs_export = os.path.join(
                self.profiler_output_tmp_dir.name, "markdown"
            )
            trace_handlers.append(
                markdown_trace_handler(markdown_logs_export, rank=self.rank)
            )

            # export stacks in txt
            stacks_logs_export = os.path.join(
                self.profiler_output_tmp_dir.name, "stacks"
            )
            stack_metrics = ["self_cpu_time_total"]
            if torch.cuda.is_available():
                stack_metrics.append("self_cuda_time_total")

            trace_handlers.append(
                export_stack_trace_handler(
                    stacks_logs_export, rank=self.rank, metrics=stack_metrics
                )
            )

            # export tensorboard
            # NOTE: removed due to segfault in pytorch 1.11.0
            # will need to be uncommented for pytorch 1.11.1 which has a fix
            # tensorboard_logs_export = os.path.join(
            #     self.profiler_output_tmp_dir.name, "tensorboard_logs"
            # )
            # trace_handlers.append(torch.profiler.tensorboard_trace_handler(
            #     tensorboard_logs_export
            # ))

            # profiler takes 1 handler, we're composing all above in a single handler
            trace_handler = composite_trace_handler(trace_handlers)

            # process every single step
            profiler_schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)

            # initialize profiler
            self.profiler = torch.profiler.profile(
                schedule=profiler_schedule,
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
                activities=activities,
                with_stack=True,  # needed to export stacks
                on_trace_ready=trace_handler,
            )
            self.profiler.start()

        else:
            self.logger.info(f"Profiler not started (enabled=False).")
            self.profiler = None

        return self.profiler

    def stop_profiler(self) -> None:
        """Stops the pytorch profiler and logs the outputs using mlflow"""
        if self.profiler:
            self.logger.info(f"Stopping profiler.")
            self.profiler.stop()

            # log via mlflow
            self.logger.info(
                f"MLFLOW log {self.profiler_output_tmp_dir.name} as an artifact."
            )
            mlflow.log_artifacts(
                self.profiler_output_tmp_dir.name, artifact_path="profiler"
            )

            self.logger.info(
                f"Clean up profiler temp dir {self.profiler_output_tmp_dir.name}"
            )
            self.profiler_output_tmp_dir.cleanup()
        else:
            self.logger.info(
                "Not stopping profiler as it was not started in the first place."
            )


class LogTimeBlock(object):
    """This class should be used to time a code block.
    The time diff is computed from __enter__ to __exit__.
    Example
    -------
    ```python
    with LogTimeBlock("my_perf_metric_name"):
        print("(((sleeping for 1 second)))")
        time.sleep(1)
    ```
    """

    def __init__(self, name, **kwargs):
        """
        Constructs the LogTimeBlock.
        Args:
        name (str): key for the time difference (for storing as metric)
        kwargs (dict): any keyword will be added  as properties to metrics for logging (work in progress)
        """
        # kwargs
        self.step = kwargs.get("step", None)
        self.enabled = kwargs.get("enabled", True)

        # internal variables
        self.name = name
        self.start_time = None
        self._logger = logging.getLogger(__name__)

    def __enter__(self):
        """Starts the timer, gets triggered at beginning of code block"""
        if not self.enabled:
            return
        self.start_time = time.time()  # starts "timer"

    def __exit__(self, exc_type, value, traceback):
        """Stops the timer and stores accordingly
        gets triggered at beginning of code block.

        Note:
            arguments are by design for with statements.
        """
        if not self.enabled:
            return
        run_time = time.time() - self.start_time  # stops "timer"

        self._logger.info(
            f"--- time elapsed: {self.name} = {run_time:2f} s [step={self.step}]"
        )
        mlflow.log_metric(self.name + ".time", run_time)


class LogDiskIOBlock(object):
    def __init__(self, name, **kwargs):
        """
        Constructs the LogDiskUsageBlock.
        Args:
        name (str): key for the time difference (for storing as metric)
        kwargs (dict): any keyword will be added  as properties to metrics for logging (work in progress)
        """
        # kwargs
        self.step = kwargs.get("step", None)
        self.enabled = kwargs.get("enabled", True)

        # internal variables
        self.name = name
        self.process_id = os.getpid()  # focus on current process
        self.start_time = None
        self.start_disk_counters = None
        self._logger = logging.getLogger(__name__)

    def __enter__(self):
        """Get initial values, gets triggered at beginning of code block"""
        if not self.enabled:
            return
        try:
            import psutil

            self.start_time = time.time()
            self.start_disk_counters = psutil.Process(self.process_id).io_counters()

        except ModuleNotFoundError:
            self.logger.critical("import psutil failed, cannot display disk stats.")

    def __exit__(self, exc_type, value, traceback):
        """Stops the timer and stores accordingly
        gets triggered at beginning of code block.

        Note:
            arguments are by design for with statements.
        """
        if not self.enabled:
            return
        try:
            import psutil
        except ModuleNotFoundError:
            self.logger.critical("import psutil failed, cannot display disk stats.")
            return

        run_time = time.time() - self.start_time

        disk_io_metrics = {}
        end_disk_counters = psutil.Process(self.process_id).io_counters()
        disk_io_metrics[f"{self.name}.disk.read"] = (
            end_disk_counters.read_bytes - self.start_disk_counters.read_bytes
        ) / (1024 * 1024)
        disk_io_metrics[f"{self.name}.disk.write"] = (
            end_disk_counters.write_bytes - self.start_disk_counters.write_bytes
        ) / (1024 * 1024)

        self._logger.info(
            f"--- time elapsed: {self.name} = {run_time:2f} s [step={self.step}]"
        )
        self._logger.info(f"--- disk_io_metrics: {disk_io_metrics}s [step={self.step}]")

        mlflow.log_metrics(disk_io_metrics)


class LogTimeOfIterator:  # lgtm [py/iter-returns-non-self]
    """This class is intended to "wrap" an existing Iterator
    and log metrics for each next() call"""

    def __init__(
        self,
        wrapped_sequence: Any,
        name: str,
        enabled: bool = True,
        async_collector: dict = None,
    ):
        self.wrapped_sequence = wrapped_sequence
        self.wrapped_iterator = None

        # for metrics
        self.enabled = enabled
        self.name = name
        self.iterator_times = []
        self.metrics = {}
        self.async_collector = async_collector

        self._logger = logging.getLogger(__name__)

    def __iter__(self):
        """Creates the iterator"""
        if self.enabled:
            start_time = time.time()
            # if enabled, creates iterator from wrapped_sequence
            self.wrapped_iterator = self.wrapped_sequence.__iter__()
            self.metrics[f"{self.name}.init"] = time.time() - start_time

            # return self
            return self
        else:
            # if disabled, return the iterator from wrapped_sequence
            # so that LogTimeOfIterator.__next__() will never get called
            return self.wrapped_sequence.__iter__()

    def __next__(self):
        """Iterates"""
        try:
            start_time = time.time()
            next_val = self.wrapped_iterator.__next__()
            self.iterator_times.append(time.time() - start_time)
            return next_val
        except StopIteration as e:
            self.log_metrics()
            raise e

    def log_metrics(self):
        """Logs metrics once iterator is finished"""
        self.metrics[f"{self.name}.count"] = len(self.iterator_times)
        self.metrics[f"{self.name}.time.sum"] = sum(self.iterator_times)
        self.metrics[f"{self.name}.time.first"] = self.iterator_times[0]

        if self.async_collector is not None:
            self._logger.info(f"Async MLFLOW: {self.metrics}")
            for k in self.metrics:
                self.async_collector[k] = self.metrics[k]
        else:
            self._logger.info(f"MLFLOW: {self.metrics}")
            mlflow.log_metrics(self.metrics)
