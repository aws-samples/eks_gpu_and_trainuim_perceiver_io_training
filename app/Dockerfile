# Example pytorch neuron container
# To build:
#    docker build . -f Dockerfile.bert_pretrain -t neuron-container:pytorch
# To run on EC2 Inf1 instances with AWS DLAMI:
#    docker run -it --device=/dev/neuron0 neuron-container:pytorch

FROM public.ecr.aws/docker/library/ubuntu:23.10
ARG PIP="pipx"

# This arg required to stop docker build waiting for region configuration while installing tz data from ubuntu 20
ARG DEBIAN_FRONTEND=noninteractive

# Neuron repos
ARG APT_REPO=https://apt.repos.neuron.amazonaws.com
ARG PIP_REPO=https://pip.repos.neuron.amazonaws.com

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/aws/neuron/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/amazon/efa/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/amazon/efa/lib64"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/amazon/openmpi/lib64"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN until apt-get update --fix-missing; do echo "Retrying apt-get update"; sleep 1; done \
 && apt-get install -y --no-install-recommends \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean
    
RUN until apt-get update --fix-missing; do echo "Retrying apt-get update"; sleep 1; done \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    jq \
    libopencv-dev \
    libglib2.0-0 \
    #libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    wget \
    unzip \
    vim \
    zlib1g-dev \
    openssl \
    libssl-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    tk-dev \
    libffi-dev \
    libcap-dev \
    gnupg2 \
    gpg-agent \
    pciutils \
    python3-full \
    --fix-missing \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

RUN echo "deb $APT_REPO focal main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - $APT_REPO/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

ADD install-neuron.sh /install-neuron.sh
RUN /install-neuron.sh
#ADD install-nvidia.sh /install-nvidia.sh
#RUN /install-nvidia.sh
#RUN apt-get update \
# && if apt-cache show aws-neuronx-tools &>/dev/null; then \
#      apt-get install -y \
#      aws-neuronx-tools \
#      aws-neuronx-collectives \
#      aws-neuronx-runtime-lib; \
#    fi \
# && rm -rf /var/lib/apt/lists/* \
# && rm -rf /tmp/tmp* \
# && apt-get clean

WORKDIR /

RUN python3 --version
RUN until apt-get update --fix-missing; do echo "Retrying apt-get update"; sleep 1; done \
 && apt-get install -y --no-install-recommends \
    python3-pip \
    --fix-missing \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean
RUN python3 -m pip install --user pipx \
 && python3 -m pipx ensurepath

RUN ${PIP} --version
RUN ${PIP} install --no-cache-dir -U \
    "pyyaml>=5.4,<5.5" \
    "bokeh>=2.3,<3" \
    "opencv-python>=4.6,<5" \
    awscli \
    einops \
    poetry \
    tensorboard \
    scipy \
    click \
    "cryptography>3.2" \
    psutil==5.6.7 \
    dataset \
    av \
    imageio \
    "lightning>=2.0.2" \
    "fvcore>=0.1.5" \
    "torch==1.13.1" \
    "torchvision==0.14.1" \
    "transformers>=4.29.2" \
    "transformers>=4.29.2" \
    "huggingface-hub>=0.15.1" \
    "fvcore>=0.1.5" \
    'Pillow>=9.0.1,<9.1.0'

RUN mkdir -p /etc/pki/tls/certs && cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

ADD install-pip-neuron.sh /install-pip-neuron.sh
RUN /install-pip-neuron.sh
#RUN ${PIP} config set global.extra-index-url $PIP_REPO \
# && ${PIP} install --force-reinstall torch-neuronx==1.13.0.* neuronx-cc==2.* --extra-index-url $PIP_REPO

# attrs, neurox-cc required: >=19.2.0, sagemaker 2.103.0 <22,>=20.3.0
# protobuf neurox-cc<4 , sagemaker training <3.20,>=3.9.2
# awscli 1.25.47 has requirement docutils<0.17,>=0.10
# etcd for kubernetes installation
RUN ${PIP} install --no-cache-dir -U \
    'attrs>=20.3.0,<22.0.0' \
    'protobuf>=2.9.2,<3.20' \
    'docutils>=0.10,<0.17' \
    "python-etcd"

# Clean up after apt update
RUN rm -rf /var/lib/apt/lists/* \
 && rm -rf /tmp/tmp* \
 && apt-get clean


# NOTE: Preceding Docker statements build a standard Neuron-compatible container with EFA support. Add your
#   training script and dependencies below this line

# Add a basic environment test script to the container image
COPY * /
RUN mkdir /config
ADD config/main.yaml /config/main.yaml
RUN echo "export PATH=/opt/aws/neuron/bin:\$PATH" >> /root/.bashrc
RUN echo "export TERM=screen" >> /root/.bashrc
