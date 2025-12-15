#!/bin/bash

# Set the data directory
DATA=${DATA:-"../data"}
DATASET_DIR="$DATA/ucf101"

echo "Creating directory structure..."
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading UCF-101-midframes.zip..."
if command -v gdown &> /dev/null; then
    gdown 10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O
else
    echo "Warning: gdown not found. Please install it with 'pip install gdown' and run:"
    echo "gdown 10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O"
    exit 1
fi

echo "Extracting UCF-101-midframes..."
unzip UCF-101-midframes.zip

echo "Downloading split_zhou_UCF101.json..."
if command -v gdown &> /dev/null; then
    gdown 1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y
else
    echo "Warning: gdown not found. Please install it with 'pip install gdown' and run:"
    echo "gdown 1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y"
fi

echo "Cleaning up compressed files..."
rm UCF-101-midframes.zip

echo "Done! Directory structure:"
tree -L 2 "$DATASET_DIR" || ls -R "$DATASET_DIR"

echo ""
echo "Dataset downloaded to: $DATASET_DIR"
echo "Final structure:"
echo "ucf101/"
echo "├── UCF-101-midframes/"
echo "└── split_zhou_UCF101.json"
