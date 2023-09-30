#!/bin/bash -x
apt-get update 
if apt-cache show aws-neuronx-tools &>/dev/null; then 
   apt-get install -y aws-neuronx-tools aws-neuronx-collectives aws-neuronx-runtime-lib 
fi 
rm -rf /var/lib/apt/lists/* 
rm -rf /tmp/tmp* 
apt-get clean
