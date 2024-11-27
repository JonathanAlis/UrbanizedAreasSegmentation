import re
def extract_integers(string):
    # Use regular expressions to find the patterns 'x=integer' and 'y=integer' (ignoring decimal points and non-digit characters after the numbers)
    x_match = re.search(r'x=(\d+)', string)
    y_match = re.search(r'y=(\d+)', string)
    
    # Extract the numbers as integers, or return None if not found
    x_value = int(x_match.group(1)) if x_match else None
    y_value = int(y_match.group(1)) if y_match else None
    
    return x_value, y_value