# Parallelized Training 

This project demonstrates the implementation of parallelism techniques to optimize the training of a large language model (LLM) across multiple GPUs.

## Metrics
The metrics were obtained by evaluating baseline and parallelized training.

## Getting Started

1. **Install the requirements:**
	```bash
	pip install -r requirements.txt
	```

2. **Run the baseline training script:**
	```bash
	python training/baseline/train_baseline.py
	```

3. **Run the model parallel training script:**
	```bash
	python training/model_parallel/train_model_parallel.py
	```

4. **Run the pipeline parallel training script:**
	```bash
	python training/pipeline_parallel/train_pipeline_parallel.py
	```

5. **Run the data parallel training script:**
	```bash
	python training/data_parallel/train_data_parallel.py
	```

6. **Run the 3D parallel training script:**
	```bash
	python training/combined_parallel/train_3d_parallel.py
	```
## Project Roadmap
- [ ] Baseline training and metrics
- [ ] Model parallelism
- [ ] Pipeline parallelism
- [ ] Data parallelism
- [ ] 3D parallelism
