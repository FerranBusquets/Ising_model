# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:50:48 2024

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuración inicial
def initialize_spins(N):
    return np.random.choice([-1, 1], size=(N, N))

def calculate_energy(spins, J, h):
    """Calcula la energía total del sistema."""
    N = spins.shape[0]
    energy = 0
    for i in range(N):
        for j in range(N):
            S = spins[i, j]
            # Spins vecinos con condiciones de contorno periódicas
            neighbors = spins[(i+1)%N, j] + spins[i, (j+1)%N] + \
                        spins[(i-1)%N, j] + spins[i, (j-1)%N]
            energy -= J * S * neighbors
            energy -= h * S
    return energy / 2  # Evitar doble conteo

def metropolis_step(spins, T, J, h):
    """Un solo paso de Metropolis."""
    N = spins.shape[0]
    i, j = np.random.randint(0, N, size=2)  # Escoger un spin aleatoriamente
    S = spins[i, j]
    
    # Spins vecinos con condiciones de contorno periódicas
    neighbors = spins[(i+1)%N, j] + spins[i, (j+1)%N] + \
                spins[(i-1)%N, j] + spins[i, (j-1)%N]
    
    # Cambio de energía si invertimos este spin
    dE = 2 * J * S * neighbors + 2 * h * S
    
    # Aceptar el cambio con probabilidad P
    if dE < 0 or np.random.rand() < np.exp(-dE / T):
        spins[i, j] *= -1  # Invertir el spin
    
    return spins

def montecarlo_ising_animation(N, T, steps, J=1, h=0, save_every=100):
    """Simulación del modelo de Ising con animación."""
    spins = initialize_spins(N)
    snapshots = [spins.copy()]  # Guardar la configuración inicial
    for step in range(steps):
        spins = metropolis_step(spins, T, J, h)
        if step % save_every == 0:
            snapshots.append(spins.copy())  # Guardar configuración periódicamente
    return snapshots

# Parámetros
N = 20            # Tamaño de la matriz (N x N)
T = 2.5           # Temperatura
steps = 10000     # Número de pasos Monte Carlo
J = 1             # Interacción entre spins
h = 0             # Campo magnético externo
save_every = 100  # Guardar estado cada ciertos pasos

# Ejecutar la simulación
snapshots = montecarlo_ising_animation(N, T, steps, J, h, save_every)

# Crear la animación
fig, ax = plt.subplots()
im = ax.imshow(snapshots[0], cmap='coolwarm', interpolation='nearest')
ax.set_title("Evolución del sistema")

def update(frame):
    im.set_data(snapshots[frame])
    ax.set_title(f"Paso {frame * save_every}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100, blit=True)

# Guardar la animación como video
ani.save('ising_model_simulation.mp4', writer='ffmpeg')

plt.show()
