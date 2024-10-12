import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scienceplots

# Configure le style des plots
plt.style.use('science')

# Chargement des données
# ./study_cases/1_jacobi_seq_asymptotic_accuracy/results/jacobi_sequential_asymptotic_accuracy.csv
fname: str = "./study_cases/1_jacobi_seq_asymptotic_accuracy/results/jacobi_sequential_asymptotic_accuracy.csv"
# fname: str = "./results/gauss_seidel_sequential_asymptotic_accuracy.csv"
data = np.genfromtxt(fname=fname, dtype=float, delimiter='; ', names=True)

print(data[r'h2'])
epsilon_values = data['epsilon']
h_values = data['h2']
error_values = data['error']
iterations = data['last_iteration']

unique_epsilons = np.unique(epsilon_values)

# Fonction pour extraire un sous-ensemble de données selon epsilon
def extract_associated_data(epsilon):
    mask = epsilon_values == epsilon
    h_subset = h_values[mask]
    error_subset = error_values[mask]
    return h_subset, error_subset

# Définition du modèle de fit et ajustement des données
def modele(h, *p):
    return p[0] + p[1] * h

def apply_fit(modele, lst_x, lst_y, p0, names):
    popt, pcov = opt.curve_fit(modele, xdata=lst_x, ydata=lst_y, absolute_sigma=False, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    print(f"Matrice de covariance : \n {pcov}")
    print("Valeurs minimisantes :")
    for i in range(len(names)):
        print(f"{names[i]} = {popt[i]:.1} ± {perr[i]:.1}")
    return popt, pcov, perr

names = ["a_0", "a_1"]
p0 = [1, 1]
h_subset, error_subset = extract_associated_data(np.min(epsilon_values))
popt, pcov, perr = apply_fit(modele=modele, lst_x=h_subset, lst_y=error_subset, p0=p0, names=names)

# Tracé des données de précision
fig = plt.figure()
ax = fig.add_subplot()

lst_h = np.linspace(h_subset.min(), h_subset.max(), num=500)
# ax.plot(lst_h, lst_h)
fit_label = fr"{popt[0]:1.1}(\pm {perr[0]:.1}) + {popt[1]:1.1}(\pm {perr[1]:.1})h^2 + O(h^3)"
fit_line, = ax.plot(lst_h, modele(lst_h, *popt), color='orange')  # On garde une référence à la ligne du fit
fit_legend = fig.legend([fit_line], [f"Fit: ${fit_label}$"], loc='upper center')
ax.add_artist(fit_legend)  # Ajoute la légende de fit à l'axe

# Tracé des points de données
for epsilon_value in unique_epsilons:
    h_subset, error_subset = extract_associated_data(epsilon_value)
    
    ax.plot(
        h_subset*2, 
        error_subset, 
        marker='o', 
        label=fr"${epsilon_value:.1e}$", 
        markersize=3, 
        linewidth=0.5, 
        linestyle="--"
    )

# Configuration des axes et légendes
ax.set(xlabel=r"$h^2$", ylabel=r"$\left\|\boldsymbol{e}^{(l_{\text{stop}})}\right\|$")
ax.legend(title=r"$\varepsilon$ values")

# Ajout d'une légende pour le fit


# ax.grid(True)
# fig.savefig("gauss_seidel_assymptotic_accuracy.pdf")
plt.show()