import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from enum import Enum, auto

import cmasher as cmr
cmap = cmr.lavender
import scienceplots

# Configure le style des plots
plt.style.use('science')


fname = "strong_scalability_measurement.csv"
# data_sigma = np.genfromtxt(fname=fname, dtype=float, delimiter=';', names=True)
# data = np.genfromtxt(fname=fname, dtype=float, delimiter=';', names=True, skip_header=3) 
# print(data)



# Lire le fichier en ignorant les trois premières lignes (les erreurs)
# df_uncertainties = pd.read_csv(fname, sep=";", skiprows=3, header=None, )
df = pd.read_csv(fname, sep="; ", header=3, index_col=0)

# Lire manuellement les trois premières lignes pour les erreurs
with open(fname, "r") as file:
    uncertainties = {}
    for i in range(3):
        line = file.readline().strip().split(';')
        uncertainties[line[0]] = float(line[1])
        



# print(uncertainties)
# print(df.loc[list(uncertainties.keys())[0]])


# print(df.loc['T_tot'].loc[-1])
# print(df)
T_seq = df.loc['T_tot', "-1"].astype(float)
lst_T_para = df.loc['T_tot'].drop('-1').astype(float)
print(lst_T_para.values)
print(lst_T_para.index.astype(int))
print(T_seq)

def S(t_para):
    return T_seq / t_para.astype(float)

def delta_S(t_para): # uncertainty on the Speedup
    return (T_seq * uncertainties['T_tot'] / (np.sqrt(2) * t_para))
print(delta_S(lst_T_para.values))


fig = plt.figure()
ax = fig.add_subplot()
# lst_T_para.index.astype(int)



ax.errorbar(lst_T_para.index.astype(int), S(lst_T_para.values), yerr=delta_S(lst_T_para.values), markersize = 5, linewidth = 1, fmt='o', label = "$S_m$", linestyle = "--")
ax.plot(lst_T_para.index.astype(int), lst_T_para.index.astype(int), label = r"$S_\text{ideal}$")

# ax.plot(lst_T_para.index.astype(int), S(T_seq, lst_T_para.values))
ax.legend()
ax.set(xlabel = r"$P$", ylabel = "$S$")
fig.savefig("jacobi_strong_scalability.pdf")
plt.show()


# # Nommer les colonnes après lecture
# df.columns = ['Key'] + [f'Value_{i}' for i in range(1, len(df.columns))]

# # Afficher les erreurs et le DataFrame
# print("Erreurs : ", errors)
# print("\nDonnées :")

# print(df.head)

# print(df[list(uncertainties.keys())[0]])