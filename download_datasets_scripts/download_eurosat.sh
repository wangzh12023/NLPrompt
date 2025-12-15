#!/bin/bash

# Set the data directory
DATA=${DATA:-"../data"}
DATASET_DIR="$DATA/eurosat"

echo "Creating directory structure..."
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading EuroSAT dataset..."
wget http://madm.dfki.de/files/sentinel/EuroSAT.zip

echo "Extracting EuroSAT dataset..."
unzip EuroSAT.zip


echo "Cleaning up compressed files..."
rm EuroSAT.zip
