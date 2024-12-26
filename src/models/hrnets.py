import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Testing
if __name__ == "__main__":

    def print_gpu_memory(prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"{prefix} Memory Allocated: {allocated:.2f} MB")
            print(f"{prefix} Memory Reserved: {reserved:.2f} MB")
        else:
            print("CUDA is not available.")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() 

