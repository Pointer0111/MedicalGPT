#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

INPUT_DIR="/root/medical/finetune"
OUTPUT_DIR="/root/medical/finetune_sharegpt"

mkdir -p $OUTPUT_DIR

for file in $INPUT_DIR/*.json; do
    filename=$(basename "$file")
    echo "Converting $filename..."
    python convert_dataset.py \
        --in_file "$file" \
        --out_file "$OUTPUT_DIR/$filename" \
        --data_type alpaca \
        --file_type json
done

echo "All done!"
