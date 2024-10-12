import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from enum import Enum, auto

import cmasher as cmr
cmap = cmr.lavender
import scienceplots

# Configure le style des plots
plt.style.use('science')


# fname = "strong_scalability_measurement.csv"
fname = "study_cases/2_jacobi_performances/results/jacobi_weak_scalability_results.csv"
df = pd.read_csv(fname, sep="; ", header=0, engine='python')

T_1 = df.loc[0, 'Time'] # runtime of one process

# other_runtimes = df[df.index != 0]
other_runtimes = df
lst_T_para = other_runtimes['Time'].values
lst_p = other_runtimes['Processes'].values
print(lst_p)
process_counts = other_runtimes.index.values


def E(t_para, p):
    return p * T_1 / ( t_para)
    
print(E(lst_T_para, lst_p))




fig = plt.figure()
ax = fig.add_subplot()


# ax.plot(lst_p, E(lst_T_para, lst_p), label = r"$S_\text{ideal}$", linewidth = 0.5)
ax.plot(lst_p, E(lst_T_para, lst_p), label = r"$E_m$", markersize = 3, linewidth = 1, marker='o', linestyle = "--")
# ax.plot(process_counts, S(lst_T_para), label = r"$S_m$", markersize = 3, linewidth = 1, marker='o', linestyle = "--")

# ax.legend()
ax.set(xlabel = r"$P$", ylabel = "$E$")
# # plt.tight_layout()
# fig.savefig("study_cases/2_jacobi_performances/results/jacobi_weak_scalability_on_my_mac.pdf")
plt.show()
