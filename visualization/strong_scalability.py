import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from enum import Enum, auto

import cmasher as cmr
cmap = cmr.lavender
import scienceplots

# Configure le style des plots
plt.style.use('science')

file_props = dict(dtype = float, delimiter = ';', names=True)

folder: str = "study_cases/2_strong_scalability"

gs_file_name = f"{folder}/gauss_seidel_strong_scalability.csv"
jac_file_name = f"{folder}/jacobi_strong_scalability.csv"

gs_data = np.genfromtxt(fname = gs_file_name, **file_props)
jac_data = np.genfromtxt(fname = jac_file_name, **file_props)

# print(gs_data['Time'])
# print(gs_data['Processes'])

gs_T_seq = gs_data['Time'][0]
gs_T_para = gs_data['Time']

jac_T_seq = jac_data['Time'][0]
jac_T_para = jac_data['Time']
# print(gs_T_para)

lst_p = jac_data['Processes']


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
plt.show()

# # fname = "strong_scalability_measurement.csv"
# fname = "study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv"
# df = pd.read_csv(fname, sep="; ", header=0, engine='python')

# T_seq = df.loc[0, 'Time'] # runtime of the sequential implementation

# other_runtimes = df[df.index != 0]
# print(other_runtimes)
# lst_T_para = other_runtimes['Time'].values
# lst_p = other_runtimes['Processes'].values
# # process_counts = other_runtimes.index.values

# def S(t_para):
#     """
#         Compute the speedup
#     """
#     return T_seq / t_para.astype(float)


# fig = plt.figure()
# ax = fig.add_subplot()


# ax.plot(lst_p, lst_p, label = r"$S_\text{ideal}$", linewidth = 0.5)
# ax.plot(lst_p, S(lst_T_para), label = r"$S_m$", markersize = 3, linewidth = 1, marker='o', linestyle = "--")

# ax.legend()
# ax.set(xlabel = r"$P$", ylabel = "$S$")
# # plt.tight_layout()
# # fig.savefig("study_cases/2_jacobi_performances/results/jacobi_strong_scalability_on_my_mac.pdf")
# plt.show()
