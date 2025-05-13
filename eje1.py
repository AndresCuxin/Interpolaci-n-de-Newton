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

# Datos del ejercicio 1
F = np.array([50, 100, 150, 200])
eps = np.array([0.12, 0.35, 0.65, 1.05])
x_vals = np.linspace(min(F), max(F), 100)
y_interp = newton_interpolation(F, eps, x_vals)
deformacion_125 = newton_interpolation(F, eps, np.array([125]))[0]
print(f"Deformación estimada para 125 N: {deformacion_125:.4f} mm")

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(F, eps, 'ro', label='Datos experimentales')
plt.plot(x_vals, y_interp, 'b-', label='Interpolación de Newton')
plt.xlabel('Fuerza (N)')
plt.ylabel('Deformación (mm)')
plt.title('Interpolación de Newton - Deformación vs. Fuerza')
plt.legend()
plt.grid(True)
plt.savefig("ejercicio1_newton.png")
plt.show()
