"""
Tests running the model loader for every possible model in the list
"""
import pytest
import torch

# local imports
from model import (
    MODEL_ARCH_LIST,
    get_model_metadata,
    load_model,
)

# IMPORTANT: see conftest.py for fixtures


@pytest.mark.parametrize("model_arch", MODEL_ARCH_LIST)
def test_model_loader(model_arch):
    """Tests src/components/pytorch_image_classifier/model/"""
    model_metadata = get_model_metadata(model_arch)

    assert model_metadata is not None
    assert isinstance(model_metadata, dict)
    assert "library" in model_metadata
    assert "input_size" in model_metadata

    # using pretrained=False to avoid downloading each time we unit test
    model = load_model(model_arch, output_dimension=4, pretrained=False)

    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_loader_failure():
    """Test asking for a model that deosn't exist"""
    with pytest.raises(NotImplementedError):
        get_model_metadata("not_a_model")

    with pytest.raises(NotImplementedError):
        load_model("not_a_model", output_dimension=4, pretrained=False)
