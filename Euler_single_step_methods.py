import numpy as np
from Newton_Raphson import newton_raphson

## Euler's explicit method

def euler_explicit(f, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32):
    h = (b - a)/N
    x_n = np.array([a + n*h for n in range(N+1)], dtype=np.float32)
    y_n = np.zeros((N+1,), dtype=np.float32)
    y_n[0] = y_0
    for n in range(N):
        y_n[n + 1] = np.round(y_n[n] + h * f(x_n[n], y_n[n]), 5)
    
    return x_n, y_n

## Euler's implicit method

def euler_implicit(f, f_y, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32):
    h = (b - a)/N
    x_n = np.array([a + n*h for n in range(N+1)], dtype=np.float32)
    y_n = np.zeros((N+1,), dtype=np.float32)
    y_n[0] = y_0
    for n in range(N):
        g = lambda y: y - y_n[n] - h*f(x_n[n+1], y)
        g_y = lambda y: 1 - h*f_y(x_n[n+1], y)
        # get y_(n+1) by solving g(y_(n+1)) = 0
        y_n[n+1] = np.round(newton_raphson(g, g_y, 0), 5)
    
    return x_n, y_n