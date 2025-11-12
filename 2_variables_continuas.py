"""
Variables Aleatorias Continuas
Ejemplos con distribuciones Normal y Exponencial
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ejemplo_normal_ingresos():
    """Ejemplo 1: Distribución de ingresos mensuales"""
    print("=== EJEMPLO 1: DISTRIBUCIÓN NORMAL - INGRESOS ===")
    
    media = 2500  # dólares
    desviacion = 500
    x = np.linspace(1000, 4000, 1000)
    pdf = stats.norm.pdf(x, media, desviacion)
    
    # Gráfico
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf, 'b-', linewidth=2)
    plt.fill_between(x, pdf, alpha=0.3)
    plt.title('Distribución Normal - Ingresos Mensuales')
    plt.xlabel('Ingreso mensual ($)')
    plt.ylabel('Densidad de probabilidad')
    plt.grid(True, alpha=0.3)
    
    # Función de distribución acumulada
    plt.subplot(1, 2, 2)
    cdf = stats.norm.cdf(x, media, desviacion)
    plt.plot(x, cdf, 'r-', linewidth=2)
    plt.title('Función de Distribución Acumulada')
    plt.xlabel('Ingreso mensual ($)')
    plt.ylabel('Probabilidad acumulada')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Cálculos prácticos
    percentil_25 = stats.norm.ppf(0.25, media, desviacion)
    percentil_75 = stats.norm.ppf(0.75, media, desviacion)
    prob_ingreso_3000 = 1 - stats.norm.cdf(3000, media, desviacion)
    prob_entre_2000_3000 = stats.norm.cdf(3000, media, desviacion) - stats.norm.cdf(2000, media, desviacion)
    
    print(f"25% gana menos de: ${percentil_25:.2f}")
    print(f"75% gana menos de: ${percentil_75:.2f}")
    print(f"Probabilidad de ganar más de $3000: {prob_ingreso_3000:.4f}")
    print(f"Probabilidad de ganar entre $2000-$3000: {prob_entre_2000_3000:.4f}")
    
    return pdf

def ejemplo_exponencial_tiempos():
    """Ejemplo 2: Tiempos de espera en servicio al cliente"""
    print("\n=== EJEMPLO 2: DISTRIBUCIÓN EXPONENCIAL - TIEMPOS DE ESPERA ===")
    
    beta = 8  # tiempo promedio de espera en minutos
    x = np.linspace(0, 25, 1000)
    pdf = stats.expon.pdf(x, scale=beta)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, pdf, 'g-', linewidth=2)
    plt.fill_between(x, pdf, alpha=0.3)
    plt.title('Distribución Exponencial - Tiempos de Espera')
    plt.xlabel('Tiempo de espera (minutos)')
    plt.ylabel('Densidad de probabilidad')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Cálculos prácticos
    prob_menos_5min = stats.expon.cdf(5, scale=beta)
    prob_mas_15min = 1 - stats.expon.cdf(15, scale=beta)
    percentil_90 = stats.expon.ppf(0.9, scale=beta)
    
    print(f"Probabilidad de esperar menos de 5 min: {prob_menos_5min:.4f}")
    print(f"Probabilidad de esperar más de 15 min: {prob_mas_15min:.4f}")
    print(f"90% de los clientes espera menos de: {percentil_90:.2f} min")
    
    return pdf

if __name__ == "__main__":
    ejemplo_normal_ingresos()
    ejemplo_exponencial_tiempos()