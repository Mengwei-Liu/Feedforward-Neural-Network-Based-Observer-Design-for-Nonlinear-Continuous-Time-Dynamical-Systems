import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp


# Define the coefficient of duffing oscillator
alpha=-1
beta=1
delta=0.3
gamma=0.37
omega=1.2

# Define the duffing oscillator function
def duffing(t,z):
    x,x_dot=z
    dxdt=x_dot
    dxdotdt=-delta*x_dot-alpha*x-beta*x**3+gamma*np.cos(omega*t)
    return [dxdt, dxdotdt]

# Start time and final time of sampling
t_span=(0,100)
# The number of sampling points
t_eval = np.linspace(t_span[0], t_span[1], 500000)

initial_conditions = [
    [0.1, 0.0],
    [1.0, 0.0],
    [-1.0, 0.5],
    [0.5, -0.5],
    [-0.5, -0.5]
]

samples = []

for z0 in initial_conditions:
    sol = solve_ivp(duffing, t_span, z0, t_eval=t_eval)
    x1 = sol.y[0]
    x2 = sol.y[1]
    for i in range(0, len(x1), 10):
        samples.append([x1[i], x2[i]])

df = pd.DataFrame(samples, columns=['x1', 'x2'])
df.to_csv("duffing_samples.csv", index=False)




