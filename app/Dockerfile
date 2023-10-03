FROM public.ecr.aws/docker/library/ubuntu:22.04

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64" 
ENV PATH="$PATH:/usr/local/cuda/bin"
ENV CUDA_HOME="$CUDA_HOME:/usr/local/cuda"

RUN uname -a
RUN apt-get update -y && apt-get install -y wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu-keyring.gpg \
 && mv cuda-wsl-ubuntu-keyring.gpg /usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | tee /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin \
 && mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600 
# && dpkg -i cuda-keyring_1.1-1_all.deb \
RUN apt-get update -y && apt-get install -y cuda

RUN ls -l /usr/local/cuda
RUN ls -l /usr/local/cuda/bin
