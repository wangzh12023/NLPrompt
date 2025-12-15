#!/bin/bash

# Download all datasets
echo "Starting to download all datasets..."

# Download DTD dataset
echo "Downloading DTD dataset..."
bash ./download_dtd.sh

# Download EuroSAT dataset
echo "Downloading EuroSAT dataset..."
bash ./download_eurosat.sh
# Download Oxford Flowers dataset
echo "Downloading Oxford Flowers dataset..."
bash ./download_oxford_flowers.sh

# Download Oxford Pets dataset
# echo "Downloading Oxford Pets dataset..."
# bash ./download_oxford_pets.sh

# Download Stanford Cars dataset
echo "Downloading Stanford Cars dataset..."
bash ./download_stanford_cars.sh

# Download UCF101 dataset
echo "Downloading UCF101 dataset..."
bash ./download_ucf101.sh

echo "All datasets downloaded successfully!"
