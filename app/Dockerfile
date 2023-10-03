FROM public.ecr.aws/docker/library/ubuntu:22.04

RUN uname -a
RUN apt-get update -y
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
