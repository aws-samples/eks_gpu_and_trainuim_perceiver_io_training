#!/bin/bash

echo "in prep_dataset.sh"

mkdir /datasets
mv /__init__.py /datasets
mv /kinetics.py /datasets

# donwload the Kinetic dataset that starts with the latter a (index 0-18)
# TODO generalize for all content - copy to FSx

mkdir -p $DATASET_DIR/train
mkdir -p $DATASET_DIR/val
mkdir -p /root/.torch/vision/datasets/kinetics/
aws s3 cp s3://deepmind-kinetics/1723303957.pt /root/.torch/vision/datasets/kinetics/
aws s3 cp s3://deepmind-kinetics/c785d04824.pt /root/.torch/vision/datasets/kinetics/
mkdir -p /var/tmp/neuron-compile-cache
aws s3 cp s3://deepmind-kinetics/var_tmp_neuron-compile-cache.tar.gz /
cd /
tar -xvzf var_tmp_neuron-compile-cache.tar.gz
cd $DATASET_DIR/train

for (( i=1;i<=9;i++ )); do wget -q "https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_00"$i".tar.gz"; tar xzf "k700_train_00"$i".tar.gz"; done

#for (( i=10;i<=18;i++ )); do wget -q "https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_0"$i".tar.gz"; tar xzf "k700_train_0"$i".tar.gz"; done

IFS=$'\n';for i in `find . -maxdepth 1 -type d`; do dir=$(echo $i | sed 's/ /_/g');mv $i $dir ;done

cd $DATASET_DIR/val

for (( i=1;i<=9;i++ )); do wget -q "https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_00"$i".tar.gz"; tar xzf "k700_val_00"$i".tar.gz"; done

#for (( i=10;i<=18;i++ )); do wget -q "https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_0"$i".tar.gz"; tar xzf "k700_val_0"$i".tar.gz"; done

IFS=$'\n';for i in `find . -maxdepth 1 -type d`; do dir=$(echo $i | sed 's/ /_/g');mv $i $dir ;done

echo "finished loading the data"
