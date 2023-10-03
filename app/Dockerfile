FROM public.ecr.aws/docker/library/ubuntu:22.04

RUN uname -a
RUN apt-get update -y && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && cp cuda-keyring_1.1-1_all.deb /usr/share/keyrings/ \
 && ls /var/
#RUN apt-get update -y && apt-get install -y cuda
