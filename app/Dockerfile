FROM public.ecr.aws/docker/library/ubuntu:20.04

RUN uname -a
RUN apt-get update -y && apt-get install -y wget python3.8-venv g++
RUN python3.8 -m pip install -U pip 
RUN pip install ipykernel 
RUN python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
RUN pip install jupyter notebook
RUN pip install environment_kernels
RUN python3.8 -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
RUN python3.8 -m pip install wget 
RUN python3.8 -m pip install awscli 
RUN python3.8 -m pip install neuronx-cc==2.* torch-neuronx torchvision
