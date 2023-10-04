FROM public.ecr.aws/docker/library/ubuntu:22.04

RUN uname -a
RUN until apt-get update --fix-missing; do echo "Retrying apt-get update"; sleep 1; done \
 && apt-get install -y --no-install-recommends \
 && wget \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean
ADD install-drivers-and-tools.sh /install-drivers-and-tools.sh 
RUN /install-drivers-and-tools.sh
ADD install-pytorch-neuron.sh /install-pytorch-neuron.sh
RUN /install-pytorch-neuron.sh
