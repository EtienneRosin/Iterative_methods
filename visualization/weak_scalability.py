
import numpy as np

def n_tilde(n, p):
    return np.sqrt(p)*(n+2) - 2


n = 10
p = 16
n_p = n_tilde(n, p)

print(f"{n = }, {n_p = }, {(n_p + 2)**2 == p *(n + 2)**2}")
print(f"{(n_p + 2)**2 = }, {(n + 2)**2 = }")
# print((n + 2)**2)
# print((n_p + 2)**2)