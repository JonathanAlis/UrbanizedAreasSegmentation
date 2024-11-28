import re
import numpy as np


def extract_integers(string):
    # Use regular expressions to find the patterns 'x=integer' and 'y=integer' (ignoring decimal points and non-digit characters after the numbers)
    x_match = re.search(r'x=(\d+)', string)
    y_match = re.search(r'y=(\d+)', string)
    
    # Extract the numbers as integers, or return None if not found
    x_value = int(x_match.group(1)) if x_match else None
    y_value = int(y_match.group(1)) if y_match else None
    
    return x_value, y_value

def unique_counts(labels):
    class_count_dict, counts = np.unique(labels, return_counts=True)
    class_count_dict = {int(class_) : int(counter_) for class_, counter_ in zip(class_count_dict, counts)}
    return class_count_dict 


class DiagonalFlip1:
    def __call__(self, tensor):
        # Diagonal Flip 1: Transpose the tensor (flip along the top-left to bottom-right diagonal)
        return tensor.permute(1, 0, *range(2, tensor.dim()))  # Swap the first two dimensions (C, H, W)

class DiagonalFlip2:
    def __call__(self, tensor):
        # Diagonal Flip 2: Flip horizontally and then transpose
        tensor = tensor.flip(2)  # Flip along the width (last dimension)
        return tensor.permute(1, 0, *range(2, tensor.dim()))  # Then transpose