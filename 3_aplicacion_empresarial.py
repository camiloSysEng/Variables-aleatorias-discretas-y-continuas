"""
Aplicación Práctica Empresarial: Optimización de Inventarios
Usando variables aleatorias discretas y continuas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class OptimizadorInventario:
    def __init__(self, demanda_media=100, desviacion_demanda=20, 
                 costo_almacenamiento=2, costo_faltante=10, lead_time=2):
        self.demanda_media = demanda_media
        self.desviacion_demanda = desviacion_demanda
        self.costo_almacenamiento = costo_almacenamiento
        self.costo_faltante = costo_faltante
        self.lead_time = lead_time
    
    def calcular_punto_reorden(self, nivel_servicio=0.95):
        """Calcula el punto de reorden usando distribución normal"""
        z_score = stats.norm.ppf(nivel_servicio)
        punto_reorden = (self.demanda_media * self.lead_time + 
                        z_score * self.desviacion_demanda * np.sqrt(self.lead_time))
        stock_seguridad = z_score * self.desviacion_demanda * np.sqrt(self.lead_time)
        return punto_reorden, stock_seguridad
    
    def simular_demanda(self, dias=1000):
        """Simula demanda usando distribución normal"""
        np.random.seed(42)  # Para reproducibilidad
        return np.random.normal(self.demanda_media, self.desviacion_demanda, dias)
    
    def optimizar_nivel_inventario(self, demanda_simulada):
        """Encuentra el nivel óptimo de inventario"""
        niveles_inventario = np.arange(80, 151, 5)
        costos_totales = []
        
        for nivel in niveles_inventario:
            costo_alm = np.maximum(nivel - demanda_simulada, 0) * self.costo_almacenamiento
            costo_falt = np.maximum(demanda_simulada - nivel, 0) * self.costo_faltante
            costo_total_promedio = np.mean(costo_alm + costo_falt)
            costos_totales.append(costo_total_promedio)
        
        nivel_optimo = niveles_inventario[np.argmin(costos_totales)]
        costo_minimo = min(costos_totales)
        
        return niveles_inventario, costos_totales, nivel_optimo, costo_minimo
    
    def analizar_sensibilidad(self):
        """Analiza sensibilidad del punto de reorden respecto al nivel de servicio"""
        niveles_servicio = np.arange(0.8, 0.991, 0.01)
        puntos_reorden = []
        
        for servicio in niveles_servicio:
            punto_reorden, _ = self.calcular_punto_reorden(servicio)
            puntos_reorden.append(punto_reorden)
        
        return niveles_servicio, puntos_reorden
    
    def ejecutar_analisis_completo(self):
        """Ejecuta el análisis completo de optimización"""
        print("=== ANALISIS DE OPTIMIZACION DE INVENTARIOS ===\n")
        
        # Cálculo del punto de reorden
        punto_reorden, stock_seguridad = self.calcular_punto_reorden()
        
        # Simulación y optimización
        demanda_simulada = self.simular_demanda()
        niveles, costos, nivel_optimo, costo_minimo = self.optimizar_nivel_inventario(demanda_simulada)
        
        # Análisis de sensibilidad
        niveles_servicio, puntos_reorden = self.analizar_sensibilidad()
        
        # Visualización
        self.graficar_resultados(demanda_simulada, niveles, costos, nivel_optimo, 
                               niveles_servicio, puntos_reorden, punto_reorden)
        
        # Resultados
        self.mostrar_resultados(punto_reorden, stock_seguridad, nivel_optimo, costo_minimo)
    
    def graficar_resultados(self, demanda_simulada, niveles, costos, nivel_optimo,
                          niveles_servicio, puntos_reorden, punto_reorden):
        """Genera gráficos con los resultados"""
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Distribución de demanda
        plt.subplot(2, 2, 1)
        plt.hist(demanda_simulada, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        x_normal = np.linspace(40, 160, 100)
        plt.plot(x_normal, stats.norm.pdf(x_normal, self.demanda_media, self.desviacion_demanda), 
                'r-', linewidth=2, label='Distribución Normal')
        plt.axvline(punto_reorden, color='green', linestyle='--', 
                   label=f'Punto reorden: {punto_reorden:.1f}')
        plt.title('Distribución de Demanda Diaria')
        plt.xlabel('Unidades demandadas')
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Optimización de costos
        plt.subplot(2, 2, 2)
        plt.plot(niveles, costos, 'bo-', linewidth=2, markersize=4)
        plt.axvline(nivel_optimo, color='red', linestyle='--', 
                   label=f'Nivel optimo: {nivel_optimo} unidades')
        plt.title('Optimización de Nivel de Inventario')
        plt.xlabel('Nivel de inventario')
        plt.ylabel('Costo total promedio ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 3: Sensibilidad del nivel de servicio
        plt.subplot(2, 2, 3)
        plt.plot(niveles_servicio, puntos_reorden, 'g-', linewidth=2)
        plt.title('Sensibilidad: Punto de Reorden vs Nivel de Servicio')
        plt.xlabel('Nivel de servicio')
        plt.ylabel('Punto de reorden (unidades)')
        plt.grid(True, alpha=0.3)
        
        # Gráfico 4: Costos de almacenamiento vs faltantes
        plt.subplot(2, 2, 4)
        costos_alm = []
        costos_falt = []
        
        for nivel in niveles:
            costo_alm = np.mean(np.maximum(nivel - demanda_simulada, 0)) * self.costo_almacenamiento
            costo_falt = np.mean(np.maximum(demanda_simulada - nivel, 0)) * self.costo_faltante
            costos_alm.append(costo_alm)
            costos_falt.append(costo_falt)
        
        plt.plot(niveles, costos_alm, 'orange', linewidth=2, label='Costo almacenamiento')
        plt.plot(niveles, costos_falt, 'purple', linewidth=2, label='Costo faltante')
        plt.axvline(nivel_optimo, color='red', linestyle='--', alpha=0.7)
        plt.title('Desglose de Costos')
        plt.xlabel('Nivel de inventario')
        plt.ylabel('Costo ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def mostrar_resultados(self, punto_reorden, stock_seguridad, nivel_optimo, costo_minimo):
        """Muestra los resultados del análisis"""
        print("=== RESULTADOS DE OPTIMIZACION ===")
        print(f"Demanda promedio: {self.demanda_media} unidades/dia")
        print(f"Desviacion estandar demanda: {self.desviacion_demanda} unidades")
        print(f"Tiempo de entrega: {self.lead_time} dias")
        print(f"Costo almacenamiento: ${self.costo_almacenamiento}/unidad/dia")
        print(f"Costo por faltante: ${self.costo_faltante}/unidad")
        print("\n--- RECOMENDACIONES ---")
        print(f"Punto de reorden: {punto_reorden:.1f} unidades")
        print(f"Stock de seguridad: {stock_seguridad:.1f} unidades")
        print(f"Nivel de inventario optimo: {nivel_optimo} unidades")
        print(f"Costo minimo promedio: ${costo_minimo:.2f} por dia")
        print(f"Nivel de servicio objetivo: 95%")

def main():
    """Función principal"""
    # Crear instancia del optimizador
    optimizador = OptimizadorInventario(
        demanda_media=100,
        desviacion_demanda=20,
        costo_almacenamiento=2,
        costo_faltante=10,
        lead_time=2
    )
    
    # Ejecutar análisis completo
    optimizador.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()