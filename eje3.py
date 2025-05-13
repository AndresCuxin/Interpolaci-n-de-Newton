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

# Datos del ejercicio 3
V = np.array([10, 20, 30, 40, 50, 60])
Cd = np.array([0.32, 0.30, 0.28, 0.27, 0.26, 0.25])
x_vals = np.linspace(min(V), max(V), 200)
y_interp = newton_interpolation(V, Cd, x_vals)

# Predicción
cd_35 = newton_interpolation(V, Cd, np.array([35]))[0]
print(f"Coeficiente de arrastre estimado para 35 m/s: {cd_35:.4f}")

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(V, Cd, 'ro', label='Datos experimentales')
plt.plot(x_vals, y_interp, 'm-', label='Interpolación de Newton')
plt.xlabel('Velocidad del aire (m/s)')
plt.ylabel('Coeficiente de arrastre (Cd)')
plt.title('Interpolación de Newton - Cd vs. Velocidad')
plt.legend()
plt.grid(True)
plt.savefig("ejercicio3_newton.png")
plt.show()
