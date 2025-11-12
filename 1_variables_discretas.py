"""
Variables Aleatorias Discretas
Ejemplos con distribuciones Poisson y Binomial
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ejemplo_poisson_clientes():
    """Ejemplo 1: Número de clientes que llegan a una tienda"""
    print("=== EJEMPLO 1: DISTRIBUCIÓN POISSON - CLIENTES ===")
    
    lambda_param = 5  # tasa promedio de llegada
    x = np.arange(0, 15)
    probabilidades = stats.poisson.pmf(x, lambda_param)
    
    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.bar(x, probabilidades, color='skyblue', alpha=0.7, edgecolor='navy')
    plt.title('Distribución Poisson - Clientes por hora')
    plt.xlabel('Número de clientes')
    plt.ylabel('Probabilidad')
    plt.grid(True, alpha=0.3)
    
    # Cálculos
    prob_menos_3 = stats.poisson.cdf(3, lambda_param)
    prob_exactamente_5 = stats.poisson.pmf(5, lambda_param)
    prob_mas_8 = 1 - stats.poisson.cdf(8, lambda_param)
    
    print(f"Probabilidad de 3 o menos clientes: {prob_menos_3:.4f}")
    print(f"Probabilidad de exactamente 5 clientes: {prob_exactamente_5:.4f}")
    print(f"Probabilidad de más de 8 clientes: {prob_mas_8:.4f}")
    
    plt.show()
    return probabilidades

def ejemplo_binomial_calidad():
    """Ejemplo 2: Control de calidad - productos defectuosos"""
    print("\n=== EJEMPLO 2: DISTRIBUCIÓN BINOMIAL - CALIDAD ===")
    
    n = 20  # tamaño del lote
    p = 0.05  # probabilidad de defectuoso
    x = np.arange(0, 11)
    probabilidades = stats.binom.pmf(x, n, p)
    
    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.bar(x, probabilidades, color='lightcoral', alpha=0.7, edgecolor='darkred')
    plt.title('Distribución Binomial - Productos Defectuosos')
    plt.xlabel('Número de productos defectuosos')
    plt.ylabel('Probabilidad')
    plt.grid(True, alpha=0.3)
    
    # Cálculos
    prob_ninguno = stats.binom.pmf(0, n, p)
    prob_uno_dos = stats.binom.cdf(2, n, p) - stats.binom.cdf(0, n, p)
    prob_mas_tres = 1 - stats.binom.cdf(3, n, p)
    
    print(f"Probabilidad de 0 defectuosos: {prob_ninguno:.4f}")
    print(f"Probabilidad de 1-2 defectuosos: {prob_uno_dos:.4f}")
    print(f"Probabilidad de más de 3 defectuosos: {prob_mas_tres:.4f}")
    
    plt.show()
    return probabilidades

if __name__ == "__main__":
    ejemplo_poisson_clientes()
    ejemplo_binomial_calidad()