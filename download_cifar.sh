#!/bin/bash


if [ ! -d "@dataset" ]; then
	mkdir dataset
fi

# download cifar-10 dataset
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P dataset

#extract archive
tar -xvzf dataset/cifar-10-python.tar.gz -C dataset

#delete archive 
rm dataset/cifar-10-python.tar.gz

#convert images from array to png extension
python extract_cifar_10.py
