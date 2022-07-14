# check release notes https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
FROM nvcr.io/nvidia/pytorch:22.04-py3

# Install dependencies missing in this container
# NOTE: container already has matplotlib==3.5.1 tqdm==4.62.0
COPY requirements.txt ./
RUN pip install -r requirements.txt

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
