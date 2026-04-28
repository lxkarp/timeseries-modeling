# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from pemd import value_distribution_emd

ts_2 = np.array([0, 0, 5, 6, 0, 5, 0, 0, 4, 0, 0, 0, 6, 0])
ts_1 = np.array([0, 0, 5, 6, 0, 5, 0, 0, 0, 8, 0, 0, 6, 0])

df = pd.DataFrame({
    'Time Series 1 (GT)': ts_1,
    'Time Series 2': ts_2
})

plt.figure(figsize=(10, 4))

line = sns.lineplot(data=df, palette=[sns.color_palette()[0], sns.color_palette()[1]])
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig("/Users/mbaro6/Desktop/output.png")
# %%
def MASE(u, f):
    return np.mean(np.abs(u - f)) 

def MSE(u, f):
    return np.mean((u - f) ** 2)/ np.mean(u ** 2)

def area_between(u, f):
    return np.sum(np.abs(u - f))

print("Wasserstein Distance:", value_distribution_emd(ts_2, ts_1))
print("Mean Absolute Seasonal Error:", MASE(ts_2, ts_1))
print("Mean Squared Error:", MSE(ts_2, ts_1))
print("Area Between:", area_between(ts_2, ts_1))
# %%
constraint_value = MASE(ts_2, ts_1)

def objective(x):
    return -area_between(x, ts_1)  # Negative because we want to maximize

def constraint(x):
    return MASE(ts_1, x) - constraint_value

initial_guess = np.zeros_like(ts_1) + 5  # Initial guess for the optimization
bounds = [(-5, 100) for _ in range(len(ts_1))]
constraint = {'type': 'eq', 'fun': constraint}

result = minimize(fun=objective, x0=initial_guess, constraints=constraint, bounds=bounds)
ts_3 = result.x

print("Optimized area between:", area_between(ts_1, ts_3))
print("MASE:", MASE(ts_1, ts_3))
print("Wasserstein Distance:", value_distribution_emd(ts_1, ts_3))

df['Time Series 3'] = ts_3
plt.figure(figsize=(10, 4))
sns.lineplot(data=df[['Time Series 1 (GT)', 'Time Series 3']], palette=[sns.color_palette()[0], sns.color_palette()[2]])
plt.xlabel('Time')
plt.ylabel('Value')
# plt.show()
plt.savefig("/Users/mbaro6/Desktop/sparse_optimized.png")
# %%
