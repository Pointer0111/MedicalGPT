#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python model_quant.py --unquantized_model_path /path/to/unquantized/model --quantized_model_output_path /path/to/save/quantized/model
