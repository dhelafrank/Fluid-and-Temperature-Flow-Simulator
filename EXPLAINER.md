## Detailed Explanation of the Code

### Overview

This project simulates the evolution of fluid flow and temperature distribution in a 2D domain over time. It solves a set of partial differential equations (PDEs) that describe the fluid dynamics and heat transfer, typically associated with natural convection phenomena, and saves the results as images.

### Step-by-Step Breakdown

#### 1. Imports and Setup

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime
import os
import sys
```

- **`time`**: Used for measuring the execution time of the script.
- **`numpy`**: Provides support for large, multi-dimensional arrays and matrices.
- **`matplotlib.pyplot`**: Used for plotting graphs and figures.
- **`scipy.integrate.solve_ivp`**: Solves initial value problems for systems of ODEs (ordinary differential equations).
- **`datetime`**: For creating timestamps.
- **`os` and `sys`**: For file and system operations.

#### 2. Timing and User Input

```python
start_time = time.time()
now = datetime.now().strftime('%Y%m%d_%H%M%S')
```

- **Timing**: Records the start time of the script.
- **Timestamp**: Creates a timestamp for filenames to avoid overwriting.

```python
input('\nThis project simulates ... Press enter to continue')
```

- **User Interaction**: Displays an introductory message and waits for the user to proceed.

#### 3. Grid and Domain Parameters

```python
nx = int(input("\nEnter number of grid points in the x direction: "))
ny = int(input("Enter number of grid points in the y direction: "))
Lx = int(input("\nEnter domain size in x axis: "))
Ly = int(input("Enter domain size in y axis: "))
```

- **Grid Points**: The number of divisions in the x and y directions.
- **Domain Size**: The physical size of the domain in the x and y directions.

```python
input(f"\n\n\nValues Summary:\nnx\t\t{nx} \nny\t\t{ny} \nLx\t\t{Lx} \nLy\t\t{Ly}\n\nNote:The higher your grid points and domain size values, the longer time it takes to plot\nPress enter to continue or Ctrl + C to cancel")
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
```

- **Grid Spacing**: The distance between adjacent grid points.

#### 4. Time Parameters and Initial Conditions

```python
t0, tf = 0, 10
nt = 100
Pr = 0.71
```

- **Time Parameters**: Initial time `t0`, final time `tf`, and the number of time points `nt`.
- **Prandtl Number**: A dimensionless number that characterizes the fluid flow.

```python
U = np.zeros((nx, ny))
V = np.zeros((nx, ny))
Theta = np.zeros((nx, ny))
Theta[:, 0] = 1
```

- **Initial Conditions**: Zero velocity (`U`, `V`) and temperature (`Theta`) fields, with a boundary condition on temperature.

```python
initial_conditions = np.concatenate([U.flatten(), V.flatten(), Theta.flatten()])
```

- **Flattening**: Converts 2D arrays into 1D arrays for solver compatibility.

#### 5. Defining the PDE System

```python
def pde_system(t, y):
    U = y[:nx*ny].reshape((nx, ny))
    V = y[nx*ny:2*nx*ny].reshape((nx, ny))
    Theta = y[2*nx*ny:].reshape((nx, ny))
```

- **Reshape**: Converts the 1D solution vector back into 2D arrays.

```python
dUdt = np.zeros_like(U)
dVdt = np.zeros_like(V)
dThetadt = np.zeros_like(Theta)
```

- **Derivatives**: Initialize arrays for the derivatives.

```python
for i in range(1, nx-1):
    for j in range(1, ny-1):
        dVdt[i, j] = - (U[i+1, j] - U[i-1, j]) / (2*dx) - (V[i, j+1] - V[i, j-1]) / (2*dy)
        dUdt[i, j] = -(U[i+1, j] - U[i-1, j]) / (2*dx) * U[i, j] - (V[i, j+1] - V[i, j-1]) / (2*dy) * U[i, j] + Theta[i, j] + (U[i, j+1] - 2*U[i, j] + U[i, j-1]) / dy**2
        dThetadt[i, j] = -(U[i+1, j] - U[i-1, j]) / (2*dx) * Theta[i, j] - (V[i, j+1] - V[i, j-1]) / (2*dy) * Theta[i, j] + (1 / Pr) * (Theta[i, j+1] - 2*Theta[i, j] + Theta[i, j-1]) / dy**2
```

- **Finite Difference Approximations**: Calculate the derivatives using central differences for interior points.

```python
U[:, 0] = 0
V[:, 0] = 0
Theta[:, 0] = 1
U[0, :] = 0
V[0, :] = 0
Theta[0, :] = 1
U[:, -1] = 0
V[:, -1] = 0
Theta[:, -1] = 0
U[-1, :] = 0
V[-1, :] = 0
Theta[-1, :] = 0
```

- **Boundary Conditions**: Apply boundary conditions for the edges of the domain.

```python
dydt = np.concatenate([dUdt.flatten(), dVdt.flatten(), dThetadt.flatten()])
return dydt
```

- **Flatten Derivatives**: Convert 2D arrays of derivatives back into a 1D array.

#### 6. Solving the PDE System

```python
t_eval = np.linspace(t0, tf, nt)
```

- **Time Points**: Generate an array of time points for evaluation.

```python
print("Starting PDE solver...")
solution = solve_ivp(pde_system, [t0, tf], initial_conditions, t_eval=t_eval, method='RK45')
print("PDE solver finished.")
```

- **Solve the System**: Use the `solve_ivp` function to solve the PDE system.

```python
U_sol = solution.y[:nx*ny].reshape((nx, ny, -1))
V_sol = solution.y[nx*ny:2*nx*ny].reshape((nx, ny, -1))
Theta_sol = solution.y[2*nx*ny:].reshape((nx, ny, -1))
```

- **Extract Solution**: Reshape the solution vectors into 3D arrays.

#### 7. Plotting and Saving Results

```python
output_dir = '../plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

- **Output Directory**: Create a directory for saving plots if it doesn't exist.

```python
print("Saving plots...")
plt.figure(figsize=(10, 6))
step = max(1, ny // 10)
for i in range(0, ny, step):  
    plt.plot(t_eval, Theta_sol[nx//2, i, :], label=f'Y = {i*dy:.2f}', color='black', linestyle='-', marker='o')
plt.xlabel('Dimensionless time τ')
plt.ylabel('Dimensionless temperature θ')
plt.title('Transient temperatures at various positions of Y for air')
plt.legend(loc='best', frameon=False)
plt.grid(True)
plot_filename = f'{output_dir}/{now}_transient_temperatures_bw.png'
plt.savefig(plot_filename)
plt.show()
```

- **Plotting**: Create and save plots of the temperature distribution over time at various positions in the domain.

#### 8. Final Steps

```python
end_time = time.time()
print(f"\nPlot saved as image:\n{plot_filename}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
sys.exit()
```

- **Timing**: Calculate and print the total execution time.
- **Exit**: Exit the script.

### Conclusion

This script provides a detailed simulation of fluid flow and heat transfer in a 2D domain, using numerical methods to solve the underlying PDEs. The results are visualized as temperature distribution plots saved as images, offering insights into the transient behavior of the system.