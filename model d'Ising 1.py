# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:36:50 2024

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuració inicial
def initialize_spins(N):
    return np.random.choice([-1, 1], size=(N, N))

def calculate_energy(spins, J, h):
    """Calcula l'energia total del sistema."""
    N = spins.shape[0]
    energy = 0
    for i in range(N):
        for j in range(N):
            S = spins[i, j]
            # Spins veïns amb condicions de contorn periòdiques
            neighbors = spins[(i+1)%N, j] + spins[i, (j+1)%N] + \
                        spins[(i-1)%N, j] + spins[i, (j-1)%N]
            energy -= J * S * neighbors
            energy -= h * S
    return energy / 2  # Evitar doble comptatge

def metropolis_step(spins, T, J, h):
    """Un sol pas de Metropolis."""
    N = spins.shape[0]
    i, j = np.random.randint(0, N, size=2)  # Escollir un spin aleatoriament
    S = spins[i, j]
    
    # Spins veïns amb condicions de contorn periòdiques
    neighbors = spins[(i+1)%N, j] + spins[i, (j+1)%N] + \
                spins[(i-1)%N, j] + spins[i, (j-1)%N]
    
    # Canvi d'energia si invertim aquest spin
    dE = 2 * J * S * neighbors + 2 * h * S
    
    # Acceptar el canvi amb probabilitat P
    if dE < 0 or np.random.rand() < np.exp(-dE / T):
        spins[i, j] *= -1  # Invertir el spin
    
    return spins

def montecarlo_ising(N, T, steps, J=1, h=0):
    """Simulació completa del model d'Ising amb Metropolis."""
    spins = initialize_spins(N)
    for step in range(steps):
        spins = metropolis_step(spins, T, J, h)
    return spins

# Paràmetres
N = 20            # Mida de la matriu (N x N)
T = 2.5           # Temperatura
steps = 10000     # Nombre de passos Monte Carlo
J = 1             # Interacció entre spins
h = 0             # Camp magnètic extern

# Executar la simulació
spins = montecarlo_ising(N, T, steps, J, h)

print("Configuració final d'spins:")
print(spins)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Configuració inicial")
plt.imshow(initialize_spins(N), cmap='coolwarm', interpolation='nearest')
plt.subplot(1, 2, 2)
plt.title("Configuració final")
plt.imshow(spins, cmap='coolwarm', interpolation='nearest')
plt.show()
