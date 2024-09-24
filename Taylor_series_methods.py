import numpy as np

# Taylor series method

def taylor_series_method(y_derivatives, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int = 5):
    h = (b - a)/N
    x_n = np.array([a + n*h for n in range(N+1)], dtype=np.float32)
    y_n = np.zeros((N+1,), dtype=np.float32)
    y_n[0] = y_0

    series_coefficients = np.ones((len(y_derivatives)+1,), dtype=np.float32)
    for i in range(1, len(y_derivatives) + 1): series_coefficients[i] = h*series_coefficients[i-1]/i

    for n in range(N):
        series_sum = series_coefficients.T @ np.array([y_n[n] if (i == 0) else y_derivatives[i-1](x_n[n], y_n[n]) for i in range(len(y_derivatives) + 1)], dtype=np.float32)
        y_n[n+1] = np.round(series_sum, decimals)
    
    return x_n, y_n