import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    sum = 0
    for xi,yi in zip(x,y):
        sum+=xi*yi

    return float(sum)