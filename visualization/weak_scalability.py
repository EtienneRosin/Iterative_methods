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




file_props = dict(dtype = float, delimiter = ';', names=True)

folder: str = "study_cases/3_weak_scalability"

gs_file_name = f"{folder}/gauss_seidel_weak_scalability_on_mac.csv"
jac_file_name = f"{folder}/jacobi_weak_scalability_on_mac.csv"

gs_data = np.genfromtxt(fname = gs_file_name, **file_props)
jac_data = np.genfromtxt(fname = jac_file_name, **file_props)


gs_T_seq = gs_data['Time'][0]
gs_T_para = gs_data['Time']

jac_T_seq = jac_data['Time'][0]
jac_T_para = jac_data['Time']


lst_p = jac_data['Processes']



def S(T_seq, T_para) -> float:
    return T_seq / T_para

def S_ideal(p) -> float:
    return p

def E(T_seq, T_para, p):
    return T_seq / (p * T_para)

# def E(T_seq, T_para, p):
#     return (T_seq) / T_para


fig = plt.figure()
ax = fig.add_subplot()

line_props = dict(linewidth = 1, linestyle = "--", marker = "o", markersize = 3)

ax.plot(lst_p, np.ones_like(lst_p), label = r"$E_\text{ideal}$", linestyle = ":", c = "blue")
ax.plot(lst_p, E(jac_T_seq, jac_T_para, lst_p), label = "Jacobi", **line_props, c = "green")
ax.plot(lst_p, E(gs_T_seq, gs_T_para, lst_p), label = "Gauss-Seidel", **line_props, c = "red")





ax.set(xlabel = r"$p$", ylabel = "$E$")
ax.legend()
# fig.savefig(fname="Figures/weak_scalability.pdf")
plt.show()