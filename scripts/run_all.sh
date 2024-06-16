#!/bin/bash

echo "Running baseline training..."
python training/baseline/train_baseline.py

echo "Running model parallel training..."
python training/model_parallel/train_model_parallel.py

echo "Running pipeline parallel training..."
python training/pipeline_parallel/train_pipeline_parallel.py

echo "Running data parallel training..."
python training/data_parallel/train_data_parallel.py

echo "Running 3D parallel training..."
python training/3d_parallel/train_3d_parallel.py