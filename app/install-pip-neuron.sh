#!/bin/bash -x

PIP="python3.8 -m pip"
if [ "$(uname -i)" = "x86_64" ]; then
  ${PIP} config set global.extra-index-url $PIP_REPO 
  ${PIP} install --force-reinstall torch-neuronx==1.13.0.* neuronx-cc==2.* --extra-index-url $PIP_REPO
fi
