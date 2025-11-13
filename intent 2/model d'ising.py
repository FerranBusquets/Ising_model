import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 20  # Lattice size (20x20)
N_steps = 1000  # Number of Monte Carlo steps
temperatures = np.linspace(1.5, 3.5, 10)  # Range of temperatures

# Initialize the spin lattice
def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

# Calculate energy of a spin configuration
def calculate_energy(lattice):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
            energy += -S * neighbors
    return energy / 2  # Each pair counted twice

# Calculate magnetization
def calculate_magnetization(lattice):
    return np.sum(lattice)

# Monte Carlo step using Metropolis criterion
def metropolis_step(lattice, beta):
    for _ in range(L**2):
        i, j = np.random.randint(0, L, size=2)
        S = lattice[i, j]
        neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
        dE = 2 * S * neighbors

        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1

# Main simulation loop
def simulate(temperatures, L, N_steps):
    magnetizations = []
    energies = []
    specific_heats = []
    susceptibilities = []

    for T in temperatures:
        beta = 1 / T
        lattice = initialize_lattice(L)
        M_list, E_list = [], []

        for step in range(N_steps):
            metropolis_step(lattice, beta)
            
            # Record data every few steps
            if step % 100 == 0:
                M_list.append(calculate_magnetization(lattice))
                E_list.append(calculate_energy(lattice))

        # Equilibrium properties
        M_array = np.array(M_list[-100:])  # Last 100 steps
        E_array = np.array(E_list[-100:])

        magnetizations.append(np.mean(np.abs(M_array)))  # Absolute magnetization
        energies.append(np.mean(E_array))
        specific_heats.append(beta**2 * np.var(E_array))  # Specific heat (variance of energy)
        susceptibilities.append(beta * np.var(M_array))  # Susceptibility (variance of magnetization)

    return magnetizations, energies, specific_heats, susceptibilities

# Run the simulation
magnetizations, energies, specific_heats, susceptibilities = simulate(temperatures, L, N_steps)

# Determine critical temperature (Tc) as the temperature with the steepest slope in magnetization
Tc_index = np.argmax(np.gradient(magnetizations, temperatures))
Tc = temperatures[Tc_index]

# Plot results
plt.figure(figsize=(16, 8))

# Magnetization vs Temperature
plt.subplot(2, 2, 1)
plt.plot(temperatures, magnetizations, 'o-', label='Magnetization')
plt.axvline(Tc, color='r', linestyle='--', label=f'Tc = {Tc:.2f}')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Temperature')
plt.legend()

# Energy vs Temperature
plt.subplot(2, 2, 2)
plt.plot(temperatures, energies, 'o-', label='Energy', color='red')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title('Energy vs Temperature')
plt.legend()

# Specific Heat vs Temperature
plt.subplot(2, 2, 3)
plt.plot(temperatures, specific_heats, 'o-', label='Specific Heat', color='green')
plt.axvline(Tc, color='r', linestyle='--')
plt.xlabel('Temperature')
plt.ylabel('Specific Heat')
plt.title('Specific Heat vs Temperature')
plt.legend()

# Susceptibility vs Temperature
plt.subplot(2, 2, 4)
plt.plot(temperatures, susceptibilities, 'o-', label='Susceptibility', color='purple')
plt.axvline(Tc, color='r', linestyle='--')
plt.xlabel('Temperature')
plt.ylabel('Susceptibility')
plt.title('Susceptibility vs Temperature')
plt.legend()

plt.tight_layout()
plt.show()

# Additional: Magnetization vs Energy
plt.figure()
plt.scatter(energies, magnetizations, color='blue', label='Magnetization vs Energy')
plt.xlabel('Energy')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Energy')
plt.legend()
plt.show()
