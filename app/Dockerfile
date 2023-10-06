FROM 133573815653.dkr.ecr.us-west-2.amazonaws.com/perceiver:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04

RUN uname -a
RUN apt-get update --fix-missing
RUN apt-get install -y wget gnupg2
ADD requirements.txt /requirements.txt

#ADD install-drivers-and-tools-neuron.sh /install-drivers-and-tools-neuron.sh 
#RUN /install-drivers-and-tools-neuron.sh

#ADD install-pytorch-neuron.sh /install-pytorch-neuron.sh
#RUN /install-pytorch-neuron.sh

#ADD install-drivers-and-tools-nvidia.sh /install-drivers-and-tools-nvidia.sh 
#RUN /install-drivers-and-tools-nvidia.sh

#ADD install-pytorch-nvidia.sh /install-pytorch-nvidia.sh
#RUN /install-pytorch-nvidia.sh
COPY * /
RUN pip install -r /requirements.txt
RUN mkdir /config
ADD config/main.yaml /config/main.yaml
RUN ls -l /
