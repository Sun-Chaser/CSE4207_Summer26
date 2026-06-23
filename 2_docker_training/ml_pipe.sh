#!/bin/bash
echo "Hello from Docker!"
python3 train.py train.csv model/trained_model
python3 infer.py model/trained_model test.csv output/prediction.json