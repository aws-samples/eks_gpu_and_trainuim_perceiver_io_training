FROM public.ecr.aws/docker/library/ubuntu:22.04

RUN uname -a
RUN apt-get update --fix-missing
RUN apt-get install -y wget gnupg2
ADD requirements.txt /requirements.txt

ADD install-drivers-and-tools-neuron.sh /install-drivers-and-tools-neuron.sh 
RUN /install-drivers-and-tools-neuron.sh

ADD install-pytorch-neuron.sh /install-pytorch-neuron.sh
RUN /install-pytorch-neuron.sh

ADD install-drivers-and-tools-nvidia.sh /install-drivers-and-tools-nvidia.sh 
RUN /install-drivers-and-tools-nvidia.sh

ADD install-pytorch-nvidia.sh /install-pytorch-nvidia.sh
RUN /install-pytorch-nvidia.sh
