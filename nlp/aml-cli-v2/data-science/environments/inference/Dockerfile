# check release notes https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
FROM nvcr.io/nvidia/pytorch:22.04-py3

# RUN conda install nvidia-apex=0.1.0 gxx_linux-64=8.5.0

# Install dependencies missing in this container
# NOTE: container already has matplotlib==3.5.1 tqdm==4.62.0
COPY requirements.txt ./
RUN pip install -r requirements.txt
