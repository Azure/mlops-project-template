# check release notes https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
FROM nvcr.io/nvidia/pytorch:22.04-py3

##############################################################################
# NCCL TESTS
##############################################################################
ENV NCCL_TESTS_TAG=v2.11.0

# NOTE: adding gencodes to support K80, M60, V100, A100
RUN mkdir /tmp/nccltests && \
    cd /tmp/nccltests && \
    git clone -b ${NCCL_TESTS_TAG} https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make \
    MPI=1 MPI_HOME=/opt/hpcx/ompi \
    NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80" \
    CUDA_HOME=/usr/local/cuda && \
    cp ./build/* /usr/local/bin && \
    rm -rf /tmp/nccltests

# Install dependencies missing in this container
# NOTE: container already has matplotlib==3.5.1 tqdm==4.62.0
COPY requirements.txt ./
RUN pip install -r requirements.txt

# RUN python -m pip install   azureml-defaults==1.41.0 \
#     mlflow==1.25.1 \
#     azureml-mlflow==1.41.0 \
#     transformers==4.18.0 \
#     psutil==5.9.0

# add ndv4-topo.xml
RUN mkdir /opt/microsoft/
ADD ./ndv4-topo.xml /opt/microsoft

# to use on A100, enable env var below in your job
# ENV NCCL_TOPO_FILE="/opt/microsoft/ndv4-topo.xml"

# adjusts the level of info from NCCL tests
ENV NCCL_DEBUG="INFO"
ENV NCCL_DEBUG_SUBSYS="GRAPH,INIT,ENV"

# Relaxed Ordering can greatly help the performance of Infiniband networks in virtualized environments.
ENV NCCL_IB_PCI_RELAXED_ORDERING="1"
ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"
ENV NCCL_SOCKET_IFNAME="eth0"
