import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from custom_colors import MSColors  # Optionnel
from utils import apply_fit  # Optionnel


plt.style.use('science')


folder = "./study_cases/1_asymptotic_accuracy"
fname_gauss_seidel = f"{folder}/gauss_seidel_sequential_asymptotic_accuracy_on_mac.csv"
fname_jacobi = f"{folder}/jacobi_sequential_asymptotic_accuracy_on_mac.csv"


data_gauss_seidel = np.genfromtxt(fname=fname_gauss_seidel, dtype=float, delimiter=';', names=True)
data_jacobi = np.genfromtxt(fname=fname_jacobi, dtype=float, delimiter=';', names=True)


epsilon_values = data_gauss_seidel['epsilon']
lst_h = data_gauss_seidel['h']

errors_gauss_seidel = data_gauss_seidel['error']
errors_jacobi = data_jacobi['error']

unique_epsilons = np.unique(epsilon_values)


def extract_associated_data(epsilon, lst_h, lst_error, epsilon_values):
    mask = epsilon_values == epsilon
    return lst_h[mask], lst_error[mask]


def modele(h, *p):
    return p[0] + p[1] * h + p[2] * (h**2)


def model_label(popt, perr):
    # return fr"{popt[0]:1.1f}(\pm {perr[0]:1.1f}) + {popt[1]:1.1f}(\pm {perr[1]:.1f})h + {popt[2]:1.1f}(\pm {perr[2]:.1f})h^2 + O(h^3)"
    return fr"{popt[0]:1.1f} + {popt[1]:1.1f}h + {popt[2]:1.1f}h^2"


p0 = [10, 100, 10]


h_subset_gauss_seidel, error_subset_gauss_seidel = extract_associated_data(np.min(epsilon_values), lst_h, errors_gauss_seidel, epsilon_values)
h_subset_jacobi, error_subset_jacobi = extract_associated_data(np.min(epsilon_values), lst_h, errors_jacobi, epsilon_values)


popt_gauss_seidel, pcov_gauss_seidel, perr_gauss_seidel = apply_fit(modele, h_subset_gauss_seidel, error_subset_gauss_seidel, p0, names=["a_0", "a_1", "a_2"])
popt_jacobi, pcov_jacobi, perr_jacobi = apply_fit(modele, h_subset_jacobi, error_subset_jacobi, p0, names=["a_0", "a_1", "a_2"])


fig, ax = plt.subplots()


fit_label_gauss_seidel = model_label(popt_gauss_seidel, perr_gauss_seidel)
fit_label_jacobi = model_label(popt_jacobi, perr_jacobi)

fit_lines_props = dict(linewidth=0.75, linestyle = ":")
lines_props = dict(marker='o', linestyle="--", markersize=3, linewidth=0.5)

for epsilon_value in unique_epsilons:
    h_subset_jacobi, error_subset_jacobi = extract_associated_data(epsilon_value, lst_h, errors_jacobi, epsilon_values)
    ax.plot(h_subset_jacobi, error_subset_jacobi, label=fr"Jacobi", color='green', **lines_props)
    # : $\varepsilon = {epsilon_value:.1e}$
    
    h_subset_gauss_seidel, error_subset_gauss_seidel = extract_associated_data(epsilon_value, lst_h, errors_gauss_seidel, epsilon_values)
    ax.plot(h_subset_gauss_seidel, error_subset_gauss_seidel, label=fr"G.S.", color='red', **lines_props)

fit_line_jacobi, = ax.plot(lst_h, modele(lst_h, *popt_jacobi), label=f"${fit_label_jacobi}$", color='green', **fit_lines_props)
fit_line_gauss_seidel, = ax.plot(lst_h, modele(lst_h, *popt_gauss_seidel), label=f"${fit_label_gauss_seidel}$", color='red', **fit_lines_props)


ax.set(xlabel=r"$h$", ylabel=r"$\left\|\boldsymbol{e}^{(l_{\text{stop}})}\right\|$",
    #    xscale="log", yscale="log"
       )

ax.legend(loc='upper center',  ncol=2)
fig.savefig(fname="Figures/accuracy.pdf")

plt.show()