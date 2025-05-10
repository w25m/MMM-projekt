import numpy as np
import matplotlib.pyplot as plt

# Parametry silnika i przekładni
R = 1.0
L = 0.5
J1 = 0.01
J2 = 0.02
n2 = 5.0
Kt = 0.1
Ke = 0.1

# Efektywna bezwładność
J_eq = J1 + J2 / n2**2

# Czas symulacji
Tmax = 5.0
dt = 0.001
N = int(Tmax / dt)
t = np.linspace(0, Tmax, N)

# === SYGNAŁ WEJŚCIOWY ===
typ_sygnalu = 'trojkatny'  # 'prostokatny', 'trojkatny', 'harmoniczny'

if typ_sygnalu == 'prostokatny':
    u = 5 * ((t % 2) < 1).astype(float)
elif typ_sygnalu == 'trojkatny':
    u = 5 * (2 * np.abs(2 * (t / 2 % 1) - 1) - 1)
elif typ_sygnalu == 'harmoniczny':
    f = 1.0  # Hz
    u = 5 * np.sin(2 * np.pi * f * t)
else:
    raise ValueError("Nieprawidłowy sygnał!")

# Inicjalizacja zmiennych
i = np.zeros(N)
omega1 = np.zeros(N)
theta1 = np.zeros(N)

# Funkcje pochodnych
def di_dt(i, omega, u):
    return (u - R * i - Ke * omega) / L

def domega_dt(i):
    return (Kt * i) / J_eq

# Symulacja metodą RK4
for k in range(N - 1):
    # --- Runge-Kutta dla i ---
    k1_i = di_dt(i[k], omega1[k], u[k])
    k2_i = di_dt(i[k] + 0.5 * dt * k1_i, omega1[k], u[k])
    k3_i = di_dt(i[k] + 0.5 * dt * k2_i, omega1[k], u[k])
    k4_i = di_dt(i[k] + dt * k3_i, omega1[k], u[k])
    i[k+1] = i[k] + (dt / 6) * (k1_i + 2*k2_i + 2*k3_i + k4_i)

    # --- Runge-Kutta dla omega1 ---
    k1_o = domega_dt(i[k])
    k2_o = domega_dt(i[k] + 0.5 * dt * k1_o)
    k3_o = domega_dt(i[k] + 0.5 * dt * k2_o)
    k4_o = domega_dt(i[k] + dt * k3_o)
    omega1[k+1] = omega1[k] + (dt / 6) * (k1_o + 2*k2_o + 2*k3_o + k4_o)

    # Całkowanie kąta
    theta1[k+1] = theta1[k] + omega1[k] * dt

# Wykresy
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, u)
plt.title(f'Sygnał: {typ_sygnalu}')
plt.ylabel("u(t) [V]")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, omega1, label='ω₁(t)')
plt.ylabel("ω₁ [rad/s]")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, i, label='i(t)', color='orange')
plt.xlabel("Czas [s]")
plt.ylabel("Prąd [A]")
plt.grid()

plt.tight_layout()
plt.show()
