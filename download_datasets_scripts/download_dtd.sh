#!/bin/bash

# Set the data directory
DATA=${DATA:-"../data"}
DATASET_DIR="$DATA/dtd"

echo "Downloading DTD dataset..."
cd "$DATA"
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

echo "Extracting DTD dataset..."
tar -xzf dtd-r1.0.1.tar.gz

cd "$DATASET_DIR"

echo "Downloading split_zhou_DescribableTextures.json..."
if command -v gdown &> /dev/null; then
    gdown 1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x
else
    echo "Warning: gdown not found. Please install it with 'pip install gdown' and run:"
    echo "gdown 1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"
fi

echo "Cleaning up compressed files..."
cd "$DATA"
rm dtd-r1.0.1.tar.gz

