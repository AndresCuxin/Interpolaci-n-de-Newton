import numpy as np
import matplotlib.pyplot as plt

# Funciones dadas
def newton_divided_diff(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def newton_interpolation(x_data, y_data, x):
    coef = newton_divided_diff(x_data, y_data)
    n = len(x_data)
    y_interp = np.zeros_like(x)
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j-1])
            term += coef[j] * product
        y_interp[i] = term
    return y_interp

# Datos del ejercicio 2
T = np.array([200, 250, 300, 350, 400])
eficiencia = np.array([30, 35, 40, 46, 53])
x_vals = np.linspace(min(T), max(T), 100)
y_interp = newton_interpolation(T, eficiencia, x_vals)

# Predicción
eficiencia_275 = newton_interpolation(T, eficiencia, np.array([275]))[0]
print(f"Eficiencia estimada para 275 °C: {eficiencia_275:.2f} %")

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(T, eficiencia, 'ro', label='Datos experimentales')
plt.plot(x_vals, y_interp, 'g-', label='Interpolación de Newton')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Eficiencia (%)')
plt.title('Interpolación de Newton - Eficiencia vs. Temperatura')
plt.legend()
plt.grid(True)
plt.savefig("ejercicio2_newton.png")
plt.show()
