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
fname = "study_cases/2_jacobi_performances/results/jacobi_strong_scalability_results.csv"
df = pd.read_csv(fname, sep="; ", header=0, engine='python')

T_seq = df.loc[0, 'Time'] # runtime of the sequential implementation

other_runtimes = df[df.index != 0]
print(other_runtimes)
lst_T_para = other_runtimes['Time'].values
lst_p = other_runtimes['Processes'].values
# process_counts = other_runtimes.index.values

def S(t_para):
    """
        Compute the speedup
    """
    return T_seq / t_para.astype(float)


fig = plt.figure()
ax = fig.add_subplot()


ax.plot(lst_p, lst_p, label = r"$S_\text{ideal}$", linewidth = 0.5)
ax.plot(lst_p, S(lst_T_para), label = r"$S_m$", markersize = 3, linewidth = 1, marker='o', linestyle = "--")

ax.legend()
ax.set(xlabel = r"$P$", ylabel = "$S$")
# plt.tight_layout()
# fig.savefig("study_cases/2_jacobi_performances/results/jacobi_strong_scalability_on_my_mac.pdf")
plt.show()
