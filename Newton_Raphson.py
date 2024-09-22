import numpy as np

## Newton-Raphson method

def newton_raphson(f, f_x, x_0: np.float32):
    while True:
        x_1 = np.float32(x_0 - f(x_0)/f_x(x_0))
        if (x_1 == x_0): break
        x_0 = x_1
    
    return x_0
