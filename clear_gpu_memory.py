import torch

def clear_gpu_memory():
	# Clear PyTorch cache
	torch.cuda.empty_cache()

	# Print GPU memory usage
	for i in range(torch.cuda.device_count()):
		print(f"Memory Usage of GPU {i}:")
		print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
		print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

if __name__ == "__main__":
	clear_gpu_memory()
