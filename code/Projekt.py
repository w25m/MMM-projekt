import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Ustawienia strony
st.title("Projekt - Metody Modelowania Matematycznego - Silnik z przekładnią")

# Sidebar — parametry
st.sidebar.header("Parametry silnika i przekładni")

R = st.number_input("R [Ω]", value=1.0)
L = st.number_input("L [H]", value=0.5)
J1 = st.number_input("J1 [kg·m²]", value=0.01)
J2 = st.number_input("J2 [kg·m²]", value=0.02)
n2 = st.number_input("n₂ [przełożenie]", value=5.0)
Kt = st.number_input("Kt [Nm/A]", value=0.1)
Ke = st.number_input("Ke [Vs/rad]", value=0.1)
U_max = st.number_input("U max [V]", value=5.0)

typ_sygnalu = st.selectbox(
    "Typ sygnału wejściowego",
    options=['prostokatny', 'trojkatny', 'harmoniczny']
)
amplituda = st.sidebar.slider("Amplituda sygnału [V]", 0.1, 20.0, 5.0, 0.1)
if typ_sygnalu == 'harmoniczny':
    czestotliwosc = st.sidebar.slider("Częstotliwość [Hz]", 0.1, 10.0, 1.0, 0.1)

# Parametry symulacji
Tmax = 5.0
dt = 0.001
N = int(Tmax / dt)
t = np.linspace(0, Tmax, N)

# Generowanie sygnału
if typ_sygnalu == 'prostokatny':
    u = amplituda * ((t % 2) < 1).astype(float)
elif typ_sygnalu == 'trojkatny':
    u = amplituda * (2 * np.abs(2 * (t / 2 % 1) - 1) - 1)
elif typ_sygnalu == 'harmoniczny':
    u = amplituda * np.sin(2 * np.pi * czestotliwosc * t)
else:
    st.error("Nieprawidłowy sygnał!")
    st.stop()

# Efektywna bezwładność
J_eq = J1 + J2 / n2**2

# Inicjalizacja zmiennych
i = np.zeros(N)
omega1 = np.zeros(N)
theta1 = np.zeros(N)

# Funkcje pochodnych
def di_dt(i, omega, u):
    return (u - R * i - Ke * omega) / L

def domega_dt(i):
    return (Kt * i) / J_eq #tutaj w liczniku moment obrotowy wału

# Symulacja metodą RK4
for k in range(N - 1):
    k1_i = di_dt(i[k], omega1[k], u[k])
    k2_i = di_dt(i[k] + 0.5 * dt * k1_i, omega1[k], u[k])
    k3_i = di_dt(i[k] + 0.5 * dt * k2_i, omega1[k], u[k])
    k4_i = di_dt(i[k] + dt * k3_i, omega1[k], u[k])
    i[k+1] = i[k] + (dt / 6) * (k1_i + 2*k2_i + 2*k3_i + k4_i)

    k1_o = domega_dt(i[k])
    k2_o = domega_dt(i[k] + 0.5 * dt * k1_o)
    k3_o = domega_dt(i[k] + 0.5 * dt * k2_o)
    k4_o = domega_dt(i[k] + dt * k3_o)
    omega1[k+1] = omega1[k] + (dt / 6) * (k1_o + 2*k2_o + 2*k3_o + k4_o)

#metoda eulera dla thety - przy małych krokach dokładna
    theta1[k+1] = theta1[k] + omega1[k] * dt

# Wykresy
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(t, u)
axs[0].set_title(f'Sygnał wejściowy: {typ_sygnalu}')
axs[0].set_ylabel("u(t) [V]")
axs[0].grid()

axs[1].plot(t, omega1)
axs[1].set_ylabel("ω₁(t) [rad/s]")
axs[1].grid()

axs[2].plot(t, i, color='orange')
axs[2].set_xlabel("Czas [s]")
axs[2].set_ylabel("i(t) [A]")
axs[2].grid()

plt.tight_layout()
st.pyplot(fig)

