#!/bin/bash

# Set the data directory
DATA=${DATA:-"../data"}
DATASET_DIR="$DATA/stanford_cars"

echo "Creating directory structure..."
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading train images..."
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz

echo "Downloading test images..."
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz

echo "Downloading train labels (devkit)..."
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz

echo "Downloading test labels..."
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat

echo "Extracting train images..."
tar -xzf cars_train.tgz

echo "Extracting test images..."
tar -xzf cars_test.tgz

echo "Extracting devkit..."
tar -xzf car_devkit.tgz
