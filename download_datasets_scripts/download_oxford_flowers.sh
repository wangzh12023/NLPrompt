#!/bin/bash

# Set the data directory
DATA=${DATA:-"../data"}
DATASET_DIR="$DATA/oxford_flowers"

echo "Creating directory structure..."
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading images..."
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

echo "Downloading labels..."
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

echo "Extracting images..."
tar -xzf 102flowers.tgz



echo "Cleaning up compressed files..."
rm 102flowers.tgz
