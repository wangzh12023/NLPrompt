#!/bin/bash

# Set the data directory
DATA=${DATA:-"../data"}
DATASET_DIR="$DATA/oxford_pets"

echo "Creating directory structure..."
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading images..."
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz

echo "Downloading annotations..."
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

echo "Extracting images..."
tar -xzf images.tar.gz

echo "Extracting annotations..."
tar -xzf annotations.tar.gz


echo "Cleaning up compressed files..."
rm images.tar.gz annotations.tar.gz
