import numpy as np
import matplotlib.pyplot as plt

def I(i, j, nx):
    return i + j * (nx + 2)

class JacobiSequential:
    def __init__(
        self,
        nx, ny, x_min, x_max, y_min, y_max, 
        left_condition_func, right_condition_func, 
        top_condition_func, bottom_condition_func, 
        source_func, init_guess_func, 
        exact_sol_func = None) -> None:
        
        self.nx = nx
        self.ny = ny
        self.x_min = x_min
        self.y_min = y_min
        self.dx = (x_max-x_min)/(nx + 1)
        self.dy = (y_max-y_min)/(ny + 1)
        
        self.total_points = (nx + 2)*(ny + 2)
        
        self.sol = np.zeros(self.total_points)
        self.new_sol = np.zeros(self.total_points)
        self.source = np.zeros(self.total_points)
        self.exact_sol = None if exact_sol_func is None else np.zeros(self.total_points)
        
        
        self.initialize_fields(left_condition_func, right_condition_func, 
        top_condition_func, bottom_condition_func, 
        source_func, init_guess_func, 
        exact_sol_func)
        
        
        self.r_0 = np.sqrt(self.compute_initial_squared_residual(self.sol, self.source))
        print(f"{self.r_0 = }")
        # self.r_0 = np.linalg.norm(self.source)
        
        # self.compute_one_step(self.sol, newsol=self.new_sol, source=self.source)
        pass
    
    def initialize_fields(
        self, 
        left_condition_func, right_condition_func, 
        top_condition_func, bottom_condition_func, 
        source_func, init_guess_func, 
        exact_sol_func = None):
        
        for i in range(0, self.nx + 1 + 1):
            x = self.x_min + i * self.dx
            for j in range(0, self.ny + 1 + 1):
                y = self.y_min + j * self.dy
                if (i == 0):
                    self.sol[I(i,j,self.nx)] = self.new_sol[I(i,j,self.nx)] = left_condition_func(x)
                elif (i == self.nx + 1):
                    self.sol[I(i,j,self.nx)] = self.new_sol[I(i,j,self.nx)] = right_condition_func(x)
                elif (j == 0):
                    self.sol[I(i,j,self.nx)] = self.new_sol[I(i,j,self.nx)] = bottom_condition_func(y)
                elif (j == self.ny + 1):
                    self.sol[I(i,j,self.nx)] = self.new_sol[I(i,j,self.nx)] = top_condition_func(y)
                else :
                    # self.sol[I(i,j,self.nx)] = self.new_sol[I(i,j,self.nx)] = init_guess_func(x, y)
                    self.sol[I(i,j,self.nx)] = init_guess_func(x, y)
                self.source[I(i,j,self.nx)] = source_func(x, y)
                if not (exact_sol_func is None):
                    self.exact_sol[I(i,j,self.nx)] = exact_sol_func(x,y)
                    
    
    def A_x(self, sol, i, j):
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        return (sol[I(i + 1, j,self.nx)] - 2. * sol[I(i, j,self.nx)] + sol[I(i - 1, j,self.nx)]) / dx2 + (sol[I(i, j + 1,self.nx)] - 2. * sol[I(i, j,self.nx)] + sol[I(i, j - 1,self.nx)]) / dy2
    
    def compute_initial_squared_residual(self, sol, source):
        value = 0.0
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                # tmp = source[I(i,j,self.nx)] - ((sol[I(i + 1, j,self.nx)] - 2. * sol[I(i, j,self.nx)] + sol[I(i - 1, j,self.nx)]) / dx2 + (sol[I(i, j + 1,self.nx)] - 2. * sol[I(i, j,self.nx)] + sol[I(i, j - 1,self.nx)]) / dy2)
                tmp = source[I(i,j,self.nx)] - self.A_x(sol, i, j)
                
                # tmp = -((tmp_sol[I(i + 1, j,self.nx)] + tmp_sol[I(i - 1, j,self.nx)]) / dx2 + (tmp_sol[I(i, j + 1,self.nx)]  + tmp_sol[I(i, j - 1,self.nx)]) / dy2)
                value += tmp * tmp
        return value
    
    def compute_squared_residual(self, sol, newsol, source):
        value = 0.0
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        tmp_sol = newsol - sol
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                # tmp = source[I(i,j,self.nx)] - ((sol[I(i + 1, j,self.nx)] - 2. * sol[I(i, j,self.nx)] + sol[I(i - 1, j,self.nx)]) / dx2 + (sol[I(i, j + 1,self.nx)] - 2. * sol[I(i, j,self.nx)] + sol[I(i, j - 1,self.nx)]) / dy2)
                tmp = source[I(i,j,self.nx)] - self.A_x(sol, i, j)
                # tmp = -((tmp_sol[I(i + 1, j,self.nx)] + tmp_sol[I(i - 1, j,self.nx)]) / dx2 + (tmp_sol[I(i, j + 1,self.nx)]  + tmp_sol[I(i, j - 1,self.nx)]) / dy2)
                value += tmp * tmp
        return value
    
    def compute_one_step(self, sol, newsol, source):
        value = 0.0
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        A_ii_minus_one = - 0.5 * (dx2 * dy2) / (dx2 + dy2)
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                newsol[I(i,j,self.nx)] =  source[I(i,j,self.nx)] - ((sol[I(i + 1, j,self.nx)] + sol[I(i - 1, j,self.nx)]) / dx2 + (sol[I(i, j + 1,self.nx)] + sol[I(i, j - 1,self.nx)]) / dy2)
                newsol[I(i,j,self.nx)] *= A_ii_minus_one
        return newsol
    
    def solve(self, l_max = int(1e4), epsilon = 1e-6):
        for l in range(1, l_max + 1):
            self.compute_one_step(self.sol, self.new_sol, self.source)
            
            
            r_ = np.sqrt(self.compute_squared_residual(self.sol, self.new_sol, self.source))
            self.sol = self.new_sol
            print(f"{l = }, {r_ / self.r_0 = }")
            
            
            if r_ <= epsilon * self.r_0 :
                print(f"Converged after {l} iterations")
                break
        if not (self.exact_sol is None):
                print(np.linalg.norm(self.exact_sol - self.sol))
        return self.sol
    
    def get_array_on_domain(self, field):
        
        # field_on_domain = np.array((3, self.nx + 2, self.ny + 2))
        field_on_domain = []
        for i in range(0, self.nx + 1 + 1):
            x = self.x_min + i * self.dx
            for j in range(0, self.ny + 1 + 1):
                y = self.y_min + j * self.dy
                field_on_domain.append([x, y, field[I(i,j,self.nx)]])
                # print(field_on_domain[I(i,j,self.nx)].shape)
        return np.asarray(field_on_domain)
if __name__ == '__main__':
    
    a = 1
    b = 1

    k_x = 2. * np.pi / a
    k_y = 2. * np.pi / b

    k_x_squared = k_x * k_x
    k_y_squared = k_y * k_y

    U_0 = 0.

    def dirichlet_condition(m):
        return U_0

    def constant(x, y):
        return U_0
    
    # def f(x, y):
    #     return 2. * (-a * x + x * x - b * y + y * y)
    

    def sin_sin(x, y) :
        return np.sin(k_x * x) * np.sin(k_y * y)
    

    # def u(x, y):
    #     return x * y * (a - x) * (b - y)
    
    def laplacien_sin_sin(x, y):
        return -(k_x_squared + k_y_squared) * sin_sin(x, y)
    
    #-----------------------------------
    n = 100
    nx = n
    ny = n
    x_min = y_min = 0
    
    x_max = a
    y_max = b
    
    
    
    jac = JacobiSequential(nx, ny, x_min, x_max, y_min, y_max, dirichlet_condition, dirichlet_condition, dirichlet_condition, dirichlet_condition, laplacien_sin_sin, constant, sin_sin)
    sol = jac.solve(epsilon=1e-4)
    
    # source = jac.get_array_on_domain(jac.source)
    # initial_guess = jac.get_array_on_domain(jac.sol)
    exact_sol = jac.get_array_on_domain(jac.exact_sol)
    # new_sol = jac.get_array_on_domain(jac.new_sol)
    # # print(f"{source.shape = }")
    last_sol = jac.get_array_on_domain(sol)
    # print(f"{jac.compute_squared_residual(jac.sol, jac.source) = }")
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # # ax.scatter(*last_sol.T, s = 2, label = "source")
    # # ax.scatter(*source.T, s = 2, label = "source")
    # # ax.scatter(*initial_guess.T, s = 2, label = "initial_guess")
    # # ax.scatter(*exact_sol.T, s = 2, label = "exact_sol")
    # diff = exact_sol[:,-1] - last_sol[:, -1]
    # ax.scatter(exact_sol[:, 0], exact_sol[:, 1], diff, s = 2)
    # # ax.scatter(*new_sol.T, s = 2)
    
    # # ax.legend()
    # plt.show()
    
     
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*last_sol.T, s = 2, label = "source")
    # ax.scatter(*source.T, s = 2, label = "source")
    # ax.scatter(*initial_guess.T, s = 2, label = "initial_guess")
    # ax.scatter(*exact_sol.T, s = 2, label = "exact_sol")
    # diff = exact_sol[:,-1] - last_sol[:, -1]
    # ax.scatter(exact_sol[:, 0], exact_sol[:, 1], diff)
    # ax.scatter(*new_sol.T, s = 2)
    
    # ax.legend()
    plt.show()
    pass