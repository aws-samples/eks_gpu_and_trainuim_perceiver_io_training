#!/bin/bash -x

PIP="pip"
if [ "$(uname -i)" = "x86_64" ]; then
  ${PIP} config set global.extra-index-url $PIP_REPO 
  #${PIP} install --force-reinstall torch-neuronx neuronx-cc --extra-index-url $PIP_REPO --break-system-packages
  ${PIP} install --force-reinstall torch-neuronx==1.13.0.* neuronx-cc==2.* --extra-index-url $PIP_REPO --break-system-packages
fi
