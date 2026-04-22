from typing import Dict
import numpy as np


def ensure_unique_values(inp: Dict) -> Dict:
    """Function to ensure all values in a dictionary are unique.
    Say, you have a dictionary where indices are mapped to values,
    this makes sure any new value that is duplicated gets removed
    and only the original index maps to this value.

    Ensures the returning dictionary can be inverted (as long as values are hashable)

    Args:
        dict (Dict): Input dictionary from which duplicate values have to be removed

    Returns:
        Dict: Dictionary with first occurence of key-value pair with a given value returned
    """
    res = {}
    for key, val in inp.items():
        res.setdefault(val, key)

    return dict((v, k) for k, v in res.items())


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def to_length(array, length):
    array = np.asarray(array)
    
    out_array = np.zeros(length, dtype=array.dtype)
    
    m = min(length, len(array))
    
    out_array[:m] = array[:m]
    return out_array
