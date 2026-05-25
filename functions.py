import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def s(z:np.array):
    '''Sigmoid function'''
    return 1/(1+np.exp(-z))

import numpy as np


def square(z: np.array):
    """Square signal wave function"""
    return 2 * s(100 * np.sin(np.pi * z)) - 1


