import numpy as np
from Taylor_series_methods import taylor_series_method

def P_Adams_Bashforth4_C_Adams_Moulton4(y_derivatives, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int = 5):
    if N < 4: raise ValueError("N must be >= 4")
    h = (b - a)/N
    _, y_first4 = taylor_series_method(y_derivatives, a, a + 3*h, y_0, 3, decimals)
    x_n = np.array([a + n*h for n in range(N+1)], dtype=np.float32)
    y_n = np.zeros((N+1,), dtype=np.float32)
    y_n[0 : 4] = y_first4

    f = y_derivatives[0]
    for n in range(3, N):
        f_n = f(x_n[n], y_n[n])
        f_n_1 = f(x_n[n-1], y_n[n-1])
        f_n_2 = f(x_n[n-2], y_n[n-2])
        f_n_3 = f(x_n[n-3], y_n[n-3])
        y_p = y_n[n] + h*(55*f_n - 59*f_n_1 + 37*f_n_2 - 9*f_n_3)/24
        y_p = np.round(y_p, decimals)
        y_c = y_n[n] + h*(9*f(x_n[n+1], y_p) + 19*f_n - 5*f_n_1 + 5*f_n_2)/24
        y_n[n+1] = np.round(y_c, decimals)
    
    return x_n, y_n

def P_Milne4_C_Milne_Simpson4(y_derivatives, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int = 5):
    if N < 4: raise ValueError("N must be >= 4")
    h = (b - a)/N
    _, y_first4 = taylor_series_method(y_derivatives, a, a + 3*h, y_0, 3, decimals)
    x_n = np.array([a + n*h for n in range(N+1)], dtype=np.float32)
    y_n = np.zeros((N+1,), dtype=np.float32)
    y_n[0 : 4] = y_first4

    f = y_derivatives[0]
    for n in range(3, N):
        f_n = f(x_n[n], y_n[n])
        f_n_1 = f(x_n[n-1], y_n[n-1])
        f_n_2 = f(x_n[n-2], y_n[n-2])
        y_p = y_n[n-3] + 4*h*(2*f_n - f_n_1 + 2*f_n_2)/3
        y_p = np.round(y_p, decimals)
        y_c = y_n[n-1] + h*(2*f(x_n[n+1], y_p) + 4*f_n + f_n_1)/3
        y_n[n+1] = np.round(y_c, decimals)
    
    return x_n, y_n
