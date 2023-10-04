FROM public.ecr.aws/docker/library/ubuntu:22.04

RUN uname -a
ADD install-drivers-and-tools.sh /install-drivers-and-tools.sh 
RUN /install-drivers-and-tools.sh
ADD install-pytorch-neuron.sh /install-pytorch-neuron.sh
RUN /install-pytorch-neuron.sh
