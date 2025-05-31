import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def symulujacja():
    # Pobierz dane z pól
    R = float(entry_R.get())
    L = float(entry_L.get())
    J1 = float(entry_J1.get())
    J2 = float(entry_J2.get())
    n2 = float(entry_n2.get())
    Kt = float(entry_Kt.get())
    Ke = float(entry_Ke.get())
    A = float(entry_A.get())
    U_max = float(entry_Umax.get())
    typ = signal_type.get()
    f = float(entry_f.get())
    J_eq = J1 + J2 / n2**2
    Tmax = float(entry_Tmax.get())
    duty = float(entry_duty.get()) / 100.0
    dt = 0.001
    N = int(Tmax / dt)
    t = np.linspace(0, Tmax, N)

    if typ == "prostokatny":
       u = A * ((t % (1/f)) < duty * (1/f)).astype(float)
    elif typ == "trojkatny":
        u = A * (2 * np.abs(2 * (t / 2 % 1) - 1) - 1)
    elif typ == "sinusoidalny":
        u = A * np.sin(2 * np.pi * f * t)
    else:
        print("Błąd: zły sygnał")
        return

  
    # Inicjalizacja zmiennych
    i = np.zeros(N)
    omega1 = np.zeros(N)
    theta1 = np.zeros(N)

    def di_dt(i, omega, u): return (u - R * i - Ke * omega) / L
    def domega_dt(i): return (Kt * i) / J_eq

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

        theta1[k+1] = theta1[k] + omega1[k] * dt

    # Wykresy
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(t, u)
    axs[0].set_title(f'Sygnał wejściowy: {typ}')
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
    plt.show()

# GUI
root = tk.Tk()
root.title("Symulacja silnika z przekładnią")

params = [
    ("R [Ω]", "1.0"),
    ("L [H]", "0.5"),
    ("J1 [kg·m²]", "0.01"),
    ("J2 [kg·m²]", "0.02"),
    ("n2", "5.0"),
    ("Kt", "0.1"),
    ("Ke", "0.1"),
    ("Amplituda [V]", "5.0"),
    ("U max [V]", "5.0"),
    ("Częstotliwość [Hz]", "1.0"),
    ("Czas symulacji [s]", "5.0"),
    ("Wypełnienie [%]", "50.0")  
]

entries = []

for i, (label, default) in enumerate(params):
    tk.Label(root, text=label).grid(row=i, column=0, sticky="e")
    entry = tk.Entry(root)
    entry.insert(0, default)
    entry.grid(row=i, column=1)
    entries.append(entry)

(entry_R, entry_L, entry_J1, entry_J2, entry_n2, entry_Kt, entry_Ke, entry_A, entry_Umax, entry_f, entry_Tmax, entry_duty) = entries

tk.Label(root, text="Typ sygnału").grid(row=len(params), column=0, sticky="e")
signal_type = ttk.Combobox(root, values=["prostokatny", "trojkatny", "sinusoidalny"])
signal_type.set("prostokatny")
signal_type.grid(row=len(params), column=1)

tk.Button(root, text="Symuluj", command=symulujacja).grid(row=len(params)+1, column=0, columnspan=2, pady=10)

root.mainloop()
