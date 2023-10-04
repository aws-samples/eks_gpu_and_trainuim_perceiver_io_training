FROM public.ecr.aws/docker/library/ubuntu:22.04

RUN uname -a
RUN apt-get update --fix-missing
RUN apt-get install -y wget gnupg2
ADD requirements.txt /requirements.txt
ADD install-drivers-and-tools.sh /install-drivers-and-tools.sh 
RUN /install-drivers-and-tools.sh
ADD install-pytorch-neuron.sh /install-pytorch-neuron.sh
RUN /install-pytorch-neuron.sh
ADD install-nvidia.sh /install-nvidia.sh
RUN /install-nvidia.sh

