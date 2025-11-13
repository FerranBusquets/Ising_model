# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:16:39 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuració inicial
def inicialitza_matriu(L):
    """Crea una matriu LxL amb spins inicials aleatoris."""
    return np.random.choice([-1, 1], size=(L, L))

def energia(matriu, J=1):
    """Calcula l'energia total del sistema."""
    energia_total = 0
    L = matriu.shape[0]
    for i in range(L):
        for j in range(L):
            # Spins veïns amb condicions de contorn periòdiques
            veïns = [
                matriu[i, (j-1) % L],
                matriu[i, (j+1) % L],
                matriu[(i-1) % L, j],
                matriu[(i+1) % L, j]
            ]
            energia_total -= J * matriu[i, j] * sum(veïns)
    return energia_total / 2  # Dividim entre 2 per no comptar interaccions duplicades

def magnetitzacio(matriu):
    """Calcula la magnetització total del sistema."""
    return np.sum(matriu)

# Algorisme de Montecarlo amb criteri de Metropolis
def metropolis(matriu, T, J=1):
    """Un sol pas de Montecarlo utilitzant el criteri de Metropolis."""
    L = matriu.shape[0]
    for _ in range(L * L):  # Intentem actualitzar cada spin una vegada de mitjana
        i, j = np.random.randint(0, L, size=2)  # Seleccionem un spin aleatori
        # Calculem el canvi d'energia si invertim aquest spin
        veïns = [
            matriu[i, (j-1) % L],
            matriu[i, (j+1) % L],
            matriu[(i-1) % L, j],
            matriu[(i+1) % L, j]
        ]
        delta_E = 2 * J * matriu[i, j] * sum(veïns)
        # Decidim si acceptem el canvi
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            matriu[i, j] *= -1  # Invertim el spin
    return matriu

# Simulació principal
def simula(L, N, T, J=1):
    """Simula el model d'Ising i retorna l'evolució de l'energia i la magnetització."""
    matriu = inicialitza_matriu(L)
    energia_vals = []
    magnetitzacio_vals = []

    for _ in range(N):
        matriu = metropolis(matriu, T, J)
        energia_vals.append(energia(matriu, J))
        magnetitzacio_vals.append(magnetitzacio(matriu))

    return matriu, energia_vals, magnetitzacio_vals

# Estudi en funció de la temperatura
def estudi_temperatures(L, N, temperatures, J=1):
    """Estudia la magnetització i energia mitjana a l'equilibri per diferents temperatures."""
    magnetitzacions = []
    energies = []

    for T in temperatures:
        _, energia_vals, magnetitzacio_vals = simula(L, N, T, J)
        magnetitzacions.append(np.mean(magnetitzacio_vals[int(N/2):]))  # Equilibri
        energies.append(np.mean(energia_vals[int(N/2):]))  # Equilibri

    return magnetitzacions, energies

# Determinació de la temperatura crítica
def determina_tc(temperatures, magnetitzacions):
    """Determina la temperatura crítica aproximada."""
    gradients = np.gradient(np.abs(magnetitzacions), np.abs(temperatures))
    return temperatures[np.argmax(gradients)]

# Aproximació de l'exponent crític beta
def calcula_exponent_critic(temperatures, magnetitzacions, Tc):
    """Aproxima l'exponent crític beta de la magnetització en funció de ΔT = Tc - T."""
    delta_T = Tc - np.array(temperatures)
    delta_T = delta_T[delta_T > 0]  # Considerem només temperatures menors que Tc
    mags = np.array(np.abs(magnetitzacions))[:len(delta_T)]

    # Ajust lineal log-log per beta
    log_delta_T = np.log(delta_T)
    log_mags = np.log(mags)
    coeficients = np.polyfit(log_delta_T, log_mags, 1)
    beta = -coeficients[0]

    plt.figure()
    plt.plot(log_delta_T, log_mags, 'o', label='Dades')
    plt.plot(log_delta_T, np.polyval(coeficients, log_delta_T), label=f'Ajust lineal (beta={beta:.2f})')
    plt.xlabel('log(ΔT)')
    plt.ylabel('log(Magnetització)')
    plt.title('Aproximació del exponent crític β')
    plt.legend()
    plt.show()

    return beta

# Evolució de l'energia mitjana i la magnetització total en funció de N
def evolucio_en_N(L, N, T, J=1):
    """Mostra l'evolució de l'energia mitjana i la magnetització total amb els passos de Montecarlo."""
    _, energia_vals, magnetitzacio_vals = simula(L, N, T, J)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(N), energia_vals, label='Energia mitjana')
    plt.title('Evolució de la energia mitjana')
    plt.xlabel('Passos de Montecarlo')
    plt.ylabel('Energia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(N), magnetitzacio_vals, label='Magnetització total', color='orange')
    plt.title('Evolució de la magnetització total')
    plt.xlabel('Passos de Montecarlo')
    plt.ylabel('Magnetització')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# Objectius opcionals avançats
def calcula_susceptibilitat(magnetitzacions, temperatures):
    """Calcula la susceptibilitat magnètica a partir de les fluctuacions."""
    return np.var(magnetitzacions) / np.mean(temperatures)

def calcula_calor_especifica(energies, temperatures):
    """Calcula la calor específica a partir de les fluctuacions."""
    return np.var(energies) / (np.mean(temperatures)**2)

def histograma_energia(L, N, T, J=1):
    """Genera un histograma de les energies de cada casella a l'equilibri."""
    matriu, _, _ = simula(L, N, T, J)
    L = matriu.shape[0]
    energies = []
    for i in range(L):
        for j in range(L):
            veïns = [
                matriu[i, (j-1) % L],
                matriu[i, (j+1) % L],
                matriu[(i-1) % L, j],
                matriu[(i+1) % L, j]
            ]
            energies.append(-J * matriu[i, j] * sum(veïns))

    plt.hist(energies, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histograma de l\'energia a T={:.2f}'.format(T))
    plt.xlabel('Energia')
    plt.ylabel('Freqüència')
    plt.show()
    
# Configuració de la simulació
L = 20  # Dimensions de la matriu
N = 1000  # Nombre de passos de simulació
temperatures = np.linspace(1.5, 3.5, 50)  # Diferents temperatures per a l'estudi

# Simulació en funció de les temperatures
magnetitzacions, energies = estudi_temperatures(L, N, temperatures)
Tc = determina_tc(temperatures, magnetitzacions)

# Resultats
print(f"Temperatura crítica aproximada: Tc = {Tc}")
print(f"Susceptibilitat: {calcula_susceptibilitat(magnetitzacions, temperatures)}")
print(f"Calor específica: {calcula_calor_especifica(energies, temperatures)}")

# Gràfics i histograma
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(temperatures, np.abs(magnetitzacions), label='Magnetització mitjana')
plt.axvline(Tc, color='red', linestyle='--', label=f'Tc ≈ {Tc:.2f}')
plt.title('Magnetització mitjana vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Magnetització mitjana')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(temperatures, energies, label='Energia mitjana', color='orange')
plt.axvline(Tc, color='red', linestyle='--', label=f'Tc ≈ {Tc:.2f}')
plt.title('Energia mitjana vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Energia mitjana')
plt.legend()

plt.tight_layout()
plt.show()

# Histograma a temperatura específica
histograma_energia(L, N, T=Tc, J=1)

# Exponent crític beta
beta = calcula_exponent_critic(temperatures, magnetitzacions, Tc)
print(f"Exponent crític beta: {beta:.2f}")

# Evolució en N
T_exemple = 2.5  # Exemple de temperatura
print(f"Evolució de l'energia i la magnetització per T = {T_exemple}")
evolucio_en_N(L, N, T=T_exemple, J=1)
