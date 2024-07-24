import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime
import os
import sys

# Start timing
start_time = time.time()
now = datetime.now().strftime('%Y%m%d_%H%M%S')  # Timestamp for filenames

input('\nThis project simulates the evolution of fluid flow and temperature distribution in a 2D domain over time.\nSpecifically, it solves a set of partial differential equations (PDEs) that describe the fluid dynamics and heat transfer,\ntypically associated with natural convection phenomena, and saves the plot as images.\n\nPress enter to continue')

# Define the grid parameters
nx = int(input("\nEnter number of grid points in the x direction: "))  # Number of grid points in the x-direction
ny = int(input("Enter number of grid points in the y direction: "))  # Number of grid points in the y direction
Lx = int(input("\nEnter domain size in x axis: "))
Ly = int(input("Enter domain size in y axis: "))
input(f"\n\n\nValues Summary:\nnx\t\t{nx} \nny\t\t{ny} \nLx\t\t{Lx} \nLy\t\t{Ly}\n\nNote:The higher your grid points and domain size values, the longer time it takes to plot\nPress enter to continue or Ctrl + C to cancel")
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Grid spacing

# Define the time parameters
t0, tf = 0, 10  # Initial and final time
nt = 100  # Number of time points
Pr = 0.71  # Prandtl number

# Define the initial condition
U = np.zeros((nx, ny))
V = np.zeros((nx, ny))
Theta = np.zeros((nx, ny))
Theta[:, 0] = 1  # Boundary condition at Y = 0

# Flatten the initial condition arrays
initial_conditions = np.concatenate([U.flatten(), V.flatten(), Theta.flatten()])

def pde_system(t, y):
    # Reshape the solution vector into 2D arrays
    U = y[:nx*ny].reshape((nx, ny))
    V = y[nx*ny:2*nx*ny].reshape((nx, ny))
    Theta = y[2*nx*ny:].reshape((nx, ny))
    
    # Initialize the derivatives
    dUdt = np.zeros_like(U)
    dVdt = np.zeros_like(V)
    dThetadt = np.zeros_like(Theta)
    
    # Apply finite difference approximations for interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Continuity equation
            dVdt[i, j] = - (U[i+1, j] - U[i-1, j]) / (2*dx) - (V[i, j+1] - V[i, j-1]) / (2*dy)
            
            # Momentum equation
            dUdt[i, j] = -(U[i+1, j] - U[i-1, j]) / (2*dx) * U[i, j] - (V[i, j+1] - V[i, j-1]) / (2*dy) * U[i, j] \
                         + Theta[i, j] + (U[i, j+1] - 2*U[i, j] + U[i, j-1]) / dy**2
            
            # Energy equation
            dThetadt[i, j] = -(U[i+1, j] - U[i-1, j]) / (2*dx) * Theta[i, j] - (V[i, j+1] - V[i, j-1]) / (2*dy) * Theta[i, j] \
                             + (1 / Pr) * (Theta[i, j+1] - 2*Theta[i, j] + Theta[i, j-1]) / dy**2
    
    # Apply boundary conditions
    U[:, 0] = 0  # X = 0
    V[:, 0] = 0  # X = 0
    Theta[:, 0] = 1  # Y = 0
    
    U[0, :] = 0  # Y = 0
    V[0, :] = 0  # Y = 0
    Theta[0, :] = 1  # Y = 0
    
    U[:, -1] = 0  # Y → ∞
    V[:, -1] = 0  # Y → ∞
    Theta[:, -1] = 0  # Y → ∞
    
    U[-1, :] = 0  # Y → ∞
    V[-1, :] = 0  # Y → ∞
    Theta[-1, :] = 0  # Y → ∞
    
    # Flatten the derivatives
    dydt = np.concatenate([dUdt.flatten(), dVdt.flatten(), dThetadt.flatten()])
    return dydt

# Time points for evaluation
t_eval = np.linspace(t0, tf, nt)

# Solve the system of PDEs
print("Starting PDE solver...")
solution = solve_ivp(pde_system, [t0, tf], initial_conditions, t_eval=t_eval, method='RK45')
print("PDE solver finished.")

# Reshape and extract the solution
U_sol = solution.y[:nx*ny].reshape((nx, ny, -1))
V_sol = solution.y[nx*ny:2*nx*ny].reshape((nx, ny, -1))
Theta_sol = solution.y[2*nx*ny:].reshape((nx, ny, -1))

# Create output directory if it doesn't exist
output_dir = '../plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot the results and save as image files
print("Saving plots...")
velPlotLocation = f'{output_dir}/{now}_velocity_u.png'
plt.figure()
plt.contourf(U_sol[:, :, -1], cmap='jet')
plt.colorbar()
plt.title('Velocity U')
plt.savefig(velPlotLocation)  # Save the plot as an image file
plt.close()  # Close the plot to free up memory

tempPlotLocation = f'{output_dir}/{now}_temperature_theta.png'
plt.figure()
plt.contourf(Theta_sol[:, :, -1], cmap='jet')
plt.colorbar()
plt.title('Temperature Theta')
plt.savefig(tempPlotLocation)  # Save the plot as an image file
plt.close()  # Close the plot to free up memory

# End timing
end_time = time.time()
print(f"\nPlots saved as images:\n{velPlotLocation}\n{tempPlotLocation}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")

# Exit the process
sys.exit()
