import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

import cmasher as cmr
cmap = cmr.lavender
import scienceplots

# Configure le style des plots
# plt.style.use('science')

class Method(Enum):
    JacobiMPI = auto()
    JacobiSequential = auto()
    GaussSeidelSequential = auto()

class Field(Enum):
    exact_solution = auto()
    initial_guess = auto()
    source_term = auto()
    last_iteration_solution = auto()

folder = "./results"
# folder = "."
# folder = "./study_cases/jacobi_sequential_asymptotic_accuracy/results"
method = Method.JacobiSequential
what_to_display = Field.last_iteration_solution


file_name = f"{folder}/{method.name}_{what_to_display.name}.csv"


def get_points_from_file(fname):
    res = np.genfromtxt(fname=fname, dtype=float, delimiter=';', names=True) 
    x = res['x']
    y = res['y']
    z = res['u']
    # for e in res:
    #     print(e)
    return x, y, z

def get_surface_from_points(x, y, z):
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    x_grid, y_grid = np.meshgrid(unique_x, unique_y)
    z_grid = np.zeros_like(x_grid)
    for i in range(len(z)):
        # Trouver les indices correspondants dans x_grid et y_grid
        xi = np.where(unique_x == x[i])[0][0]
        yi = np.where(unique_y == y[i])[0][0]
        z_grid[yi, xi] = z[i]  # Remplir z_grid
    return x_grid, y_grid, z_grid
        
        
X, Y, U_exact = get_points_from_file(f"{folder}/{method.name}_exact_solution.csv")
X, Y, U_init = get_points_from_file(f"{folder}/{method.name}_initial_guess.csv")

x_grid, y_grid, z_grid = get_surface_from_points(X, Y, U_exact - U_init)

# unique_x = np.unique(x)
# unique_y = np.unique(y)

# print(f"{len(unique_x) = }, {len(unique_y) = }")

# # Créer une grille de valeurs x et y
# x_grid, y_grid = np.meshgrid(unique_x, unique_y)

# # Initialiser une grille z correspondante
# z_grid = np.zeros_like(x_grid)

# # Remplir z_grid avec les valeurs de z
# for i in range(len(z)):
#     # Trouver les indices correspondants dans x_grid et y_grid
#     xi = np.where(unique_x == x[i])[0][0]
#     yi = np.where(unique_y == y[i])[0][0]
#     z_grid[yi, xi] = z[i]  # Remplir z_grid

# Création de la figure 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Tracer la surface
# surf = ax.plot_surface(*get_surface_from_points(*get_points_from_file(file_name)), cmap=cmap)
# print(U_exact)
# print(f"{U_exact.shape = }")
N = 12
# def I(i, j):
#     return i + j*(N + 2)
# def extract_boundary_terms(U, N):
#     boundary_terms = []
#     for i in range(0, len(U), N):  # Parcourir chaque tranche de 12 éléments
#         # Ajouter les termes de bord (1er et dernier élément de chaque tranche)
#         boundary_terms.append((X[i], Y[i], U[i]))         # Premier élément
#         boundary_terms.append((X[i + N - 1], Y[i + N - 1], U[i + N - 1]))  # Dernier élément
#     return boundary_terms

# # Extraire les termes de bord pour U_exact - U_init
# boundary_terms = extract_boundary_terms(U_exact - U_init, N)

# # Séparer les valeurs pour l'affichage
# X_borders, Y_borders, U_borders = zip(*boundary_terms)

# # Créer la figure et afficher uniquement les termes de bord
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X_borders, Y_borders, U_borders, color='r')  # Scatter plot pour les termes de bord
# ax.set(xlabel="$x$", ylabel="$y$", zlabel=r"$\tilde{u}(x,y)$")
# fig.canvas.manager.set_window_title(f"{method} - Boundary Terms")
print(f"{np.linalg.norm(z_grid) = }")
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cmap)
plt.show()

# for j in range(10):
    
#     ax.scatter(X[I(0,j):I(0,j)+1:], Y[I(0,j):I(0,j)+1:], U_init[I(0,j):I(0,j)+1:])


# ax.scatter(X, Y, U_exact - U_init)

# ax.scatter(X, Y, U_exact[:2:])

# surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cmap)

# Ajouter une barre de couleur
# # fig.colorbar(surf)
# ax.set(xlabel = "$x$", ylabel = "$y$", zlabel = r"$\tilde{u}(x,y)$")
# fig.canvas.manager.set_window_title(f"{method} - {what_to_display}")

# Ajouter des labels et un titre
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set(aspect = "equal")

# Afficher la figure
# fig.savefig(f"{method.name}_{what_to_display.name}.pdf")
plt.show()