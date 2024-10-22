import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from enum import Enum, auto

import cmasher as cmr
cmap = cmr.lavender
import scienceplots

# Configure le style des plots
plt.style.use('science')

# study_cases/3_weak_scalability/jacobi_weak_scalability.csv




file_props = dict(dtype = float, delimiter = ',', names=True)


file_name = f"chol_meas/strong_scalability_measurements.csv"
data = np.genfromtxt(fname = file_name, **file_props)
# jac_data = np.genfromtxt(fname = jac_file_name, **file_props)

# time_jacobi,time_gauss_seidel
gs_T_seq = data['time_gauss_seidel'][0]
gs_T_para = data['time_gauss_seidel']

jac_T_seq = data['time_jacobi'][0]
jac_T_para = data['time_jacobi']


lst_p = data['Processes']


# def E(T_seq, T_para, p):
#     return (T_seq) / T_para

# fig = plt.figure()
# ax = fig.add_subplot()

line_props = dict(linewidth = 1, linestyle = "--", marker = "o", markersize = 3)
scatter_props = dict(marker = "o", s = 3)
def S(T_seq, T_para) -> float:
    return T_seq / T_para

def S_ideal(p) -> float:
    return p



fig = plt.figure()
ax = fig.add_subplot()
line_props = dict(linewidth = 1, linestyle = "--", marker = "o", markersize = 3)

ax.plot(lst_p, S_ideal(lst_p), label = r"$S_\text{ideal}$", linestyle = ":", c = "blue")
ax.plot(lst_p, S(jac_T_seq, jac_T_para), label = "Jacobi", **line_props, c = "green")
ax.plot(lst_p, S(gs_T_seq, gs_T_para), label = "Gauss-Seidel", **line_props, c = "red")




ax.set(xlabel = r"$p$", ylabel = "$S$")
ax.legend()
fig.savefig(fname="Figures/strong_scalability_on_cholesky.pdf")
plt.show()