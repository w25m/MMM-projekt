import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametry silnika i przekładni
R = 1.0  # rezystancja [Ohm]
L = 0.5  # indukcyjność [H]
J1 = 0.01  # bezwładność pierwszego wału [kg*m^2]
J2 = 0.02  # bezwładność drugiego wału [kg*m^2]
n2 = 5.0  # przełożenie przekładni
Kt = 0.1  # stała momentu [Nm/A]
Ke = 0.1  # stała SEM [Vs/rad]

# Bezwładność efektywna widziana z wału 1
J_eq = J1 + J2 / n2 ** 2

# Transmitancja G(s) = Ω1(s) / U(s)
num = [Kt]
den = [J_eq * L, J_eq * R, Kt * Ke]
system = signal.TransferFunction(num, den)

# Czas symulacji
t = np.linspace(0, 5, 1000)

# Odpowiedź skokowa prędkości kątowej
t, omega1 = signal.step(system, T=t)

# U(t) - skok jednostkowy
u_t = np.ones_like(t)

# Oblicz i(t) = (u(t) - Ke * omega1(t)) przez układ RL
# G_I(s) = 1 / (R + Ls)
num_i = [1]
den_i = [L, R]
RL_sys = signal.TransferFunction(num_i, den_i)

# sygnał wejściowy do układu RL: u(t) - Ke * omega1(t)
input_i = u_t - Ke * omega1
t, i_t, _ = signal.lsim(RL_sys, U=input_i, T=t)

# Wykresy
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, omega1, label='ω₁(t)')
plt.ylabel("ω₁ [rad/s]")
plt.title("Odpowiedź układu")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, i_t, label='i(t)', color='orange')
plt.xlabel("Czas [s]")
plt.ylabel("Prąd [A]")
plt.grid(True)

plt.tight_layout()
plt.show()

