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
what_to_display = Field.source_term


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
surf = ax.plot_surface(*get_surface_from_points(*get_points_from_file(file_name)), cmap=cmap)

# Ajouter une barre de couleur
# fig.colorbar(surf)
ax.set(xlabel = "$x$", ylabel = "$y$", zlabel = r"$\tilde{u}(x,y)$")
fig.canvas.manager.set_window_title(f"{method} - {what_to_display}")

# Ajouter des labels et un titre
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set(aspect = "equal")

# Afficher la figure
# fig.savefig(f"{method.name}_{what_to_display.name}.pdf")
plt.show()