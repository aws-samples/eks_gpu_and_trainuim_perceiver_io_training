#!/bin/bash -x

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export PATH="$PATH:/usr/local/cuda/bin"
export CUDA_HOME="$CUDA_HOME:/usr/local/cuda"

if [ "$(uname -i)" = "x86_64" ]; then
  apt-get update -y && apt-get install -y wget
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu-keyring.gpg 
  mv cuda-wsl-ubuntu-keyring.gpg /usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg 
  echo "deb [signed-by=/usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | tee /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list 
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin 
  mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  apt-get update -y && apt-get install -y cuda
fi
