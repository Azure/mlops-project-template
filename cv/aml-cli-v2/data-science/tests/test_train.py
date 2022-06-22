"""
Tests running the train.py script end-to-end
on a randomly generated (small) dataset.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

import numpy as np
from PIL import Image

# local imports
import train
from model import MODEL_ARCH_LIST

# IMPORTANT: see conftest.py for fixtures (ex: temporary_dir)


@pytest.fixture()
def random_image_in_folder_classes(temporary_dir):
    image_dataset_path = os.path.join(temporary_dir, "image_in_folders")
    os.makedirs(image_dataset_path, exist_ok=False)

    n_samples = 100
    n_classes = 4

    for i in range(n_samples):
        a = np.random.rand(300, 300, 3) * 255
        im_out = Image.fromarray(a.astype("uint8")).convert("RGB")

        class_dir = "class_{}".format(i % n_classes)

        image_path = os.path.join(
            image_dataset_path, class_dir, "random_image_{}.jpg".format(i)
        )
        os.makedirs(os.path.join(image_dataset_path, class_dir), exist_ok=True)
        im_out.save(image_path)

    return image_dataset_path


# IMPORTANT: we have to restrict the list of models for unit test
# because github actions runners have 7GB RAM only and will OOM
TEST_MODEL_ARCH_LIST = [
    "test",
    "resnet18",
    "resnet34",
]

# NOTE: we only care about patching those specific mlflow methods
# to mlflow initialization conflict between tests
@patch("mlflow.end_run")  # we can have only 1 start/end per test session
@patch("mlflow.register_model")  # patched to test model name registration
@patch("mlflow.pytorch.log_model")  # patched to test model name registration
@patch("mlflow.log_params")  # patched to avoid conflict in parameters
@patch("mlflow.start_run")  # we can have only 1 start/end per test session
@pytest.mark.parametrize("model_arch", TEST_MODEL_ARCH_LIST)
def test_components_pytorch_image_classifier_single_node(
    mlflow_start_run_mock,
    mlflow_log_params_mock,
    mlflow_pytorch_log_model_mock,
    mlflow_register_model_mock,
    mlflow_end_run_mock,
    model_arch,
    temporary_dir,
    random_image_in_folder_classes,
):
    """Tests src/components/pytorch_image_classifier/train.py"""
    model_dir = os.path.join(temporary_dir, "pytorch_image_classifier_model")
    checkpoints_dir = os.path.join(
        temporary_dir, "pytorch_image_classifier_checkpoints"
    )

    # create test arguments for the script
    # fmt: off
    script_args = [
        "train.py",
        "--train_images", random_image_in_folder_classes,
        "--valid_images", random_image_in_folder_classes,  # using same data for train/valid
        "--batch_size", "16",
        "--num_workers", "0",  # single thread pre-fetching
        "--prefetch_factor", "2",  # will be discarded if num_workers=0
        "--pin_memory", "True",
        "--non_blocking", "False",
        "--model_arch", model_arch,
        "--model_arch_pretrained", "True",
        "--num_epochs", "2",
        "--model_output", model_dir,
        "--checkpoints", checkpoints_dir,
        "--register_model_as", "foo",
        "--enable_profiling", "True",
    ]
    # fmt: on

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        train.main()

    # those mlflow calls must be unique in the script
    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()

    # test all log_params calls
    for log_params_call in mlflow_log_params_mock.call_args_list:
        args, kwargs = log_params_call
        assert isinstance(args[0], dict)  # call has only 1 argument, and it's a dict

    # test model registration with mlflow.pytorch.log_model()
    log_model_calls = mlflow_pytorch_log_model_mock.call_args_list
    assert len(log_model_calls) == 1
    args, kwargs = log_model_calls[0] # unpack arguments
    assert "artifact_path" in kwargs
    assert kwargs["artifact_path"] == "final_model"
    assert "registered_model_name" in kwargs
    assert kwargs["registered_model_name"] == "foo"

    # test model registration with mlflow.register_model()
    register_model_calls = mlflow_register_model_mock.call_args_list
    assert len(register_model_calls) == 1 # call should happen only once
    args, kwargs = register_model_calls[0] # unpack arguments
    assert "model_uri" in kwargs
    assert kwargs["model_uri"].endswith("final_model")
    assert "name" in kwargs
    assert kwargs["name"] == "foo"

    # test checkpoints presence
    assert len(os.listdir(checkpoints_dir)) == 3  # 1 before training loop, + 2 epochs


@patch("mlflow.end_run")  # we can have only 1 start/end per test session
@patch("mlflow.register_model")  # patched to test model name registration
@patch("mlflow.pytorch.log_model")  # patched to test model name registration
@patch("mlflow.log_params")  # patched to avoid conflict in parameters
@patch("mlflow.start_run")  # we can have only 1 start/end per test session
@patch("torch.distributed.init_process_group")  # to avoid calling for the actual thing
@patch(
    "torch.distributed.destroy_process_group"
)  # to avoid calling for the actual thing
@patch(
    "torch.nn.parallel.DistributedDataParallel"
)  # to avoid calling for the actual thing
@pytest.mark.parametrize("backend", ["nccl", "mpi"])
def test_components_pytorch_image_classifier_second_of_two_nodes(
    torch_ddp_mock,
    torch_dist_destroy_process_group_mock,
    torch_dist_init_process_group_mock,
    mlflow_start_run_mock,
    mlflow_log_params_mock,
    mlflow_pytorch_log_model_mock,
    mlflow_register_model_mock,
    mlflow_end_run_mock,
    backend,
    temporary_dir,
    random_image_in_folder_classes,
):
    """Tests src/components/pytorch_image_classifier/train.py"""
    model_dir = os.path.join(
        temporary_dir, "pytorch_image_classifier_distributed_model"
    )

    torch_ddp_mock.side_effect = lambda model: model  # ddp would return just the model

    # create some environment variables for the backend
    if backend == "nccl":
        backend_expected_env = {
            # setup as if there were 2 nodes with 1 gpu each
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }
    elif backend == "mpi":
        backend_expected_env = {
            # setup as if there were 2 nodes with 1 gpu each
            "OMPI_COMM_WORLD_SIZE": "2",
            "OMPI_COMM_WORLD_RANK": "1",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "1",
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
        }
    else:
        raise Exception("backend {} used for testing is not implemented in script.")

    with patch.dict(os.environ, backend_expected_env, clear=False):
        # create test arguments for the script
        # fmt: off
        script_args = [
            "train.py",
            "--train_images", random_image_in_folder_classes,
            "--valid_images", random_image_in_folder_classes,  # using same data for train/valid
            "--distributed_backend", backend,
            "--batch_size", "16",
            "--num_workers", "0",  # single thread pre-fetching
            "--prefetch_factor", "2",  # will be discarded if num_workers=0
            "--pin_memory", "True",
            "--non_blocking", "False",
            "--model_arch", "resnet18",
            "--model_arch_pretrained", "True",
            "--num_epochs", "1",
            "--register_model_as", "foo",
        ]
        # fmt: on

        # replaces sys.argv with test arguments and run main
        with patch.object(sys, "argv", script_args):
            train.main()

    # those mlflow calls must be unique in the script
    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()

    mlflow_pytorch_log_model_mock.assert_not_called()  # not saving from non-head nodes
    mlflow_register_model_mock.assert_not_called()  # not registering from non-head nodes

    torch_dist_init_process_group_mock.assert_called_once()
    torch_dist_init_process_group_mock.assert_called_with(backend, rank=1, world_size=2)

    torch_dist_destroy_process_group_mock.assert_called_once()
