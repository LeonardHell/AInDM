""""
Lab 3
""" 
#Import everything
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize

# Data

assets = ["GOVT", "CORP", "EQTY", "INFR"]
scenarios = ["BOOM", "BASE", "SLOW", "RECN", "INFL"]
p = np.array([0.15, 0.40, 0.20, 0.15, 0.10]) #Probabilities for the scenarios
R = np.array([[-1, 3, 5, 9, -7], [8, 5, 3, -4, -3],
[25, 11, 2, -18, -10], [9, 6, 3, -2, 7]], dtype=float) # Return in % per year per scenario
C = np.array([[40, 35, 30, 25, 35], [120, 100, 90, 70, 100],
[160, 130, 110, 80, 120], [90, 90, 85, 80, 95]], dtype=float) # tCO2e per EUR 1m
ub = np.array([1.0, 1.0, 0.6, 0.2])
w_current = np.array([0.40, 0.20, 0.30, 0.10])

"""
Part A: 
A.1
"""
mu = R @ p # expected returns
sig = np.zeros(len(assets))
cov = np.zeros((len(assets),len(assets)))
for i in range(len(assets)):
    for j in range(len(R[i])):
        sig[i] += (R[i,j]-mu[i])**2*p[j]        
sig = np.sqrt(sig)
for i in range(len(assets)):
    for j in range(len(assets)):
        cov[i,j] = sum(p[s]*(R[i,s]-mu[i])*(R[j,s]-mu[j]) for s in range(len(p)))

#m,s way 1
m_1 = mu.T@w_current
s2_1 = w_current.T@cov@w_current

#m,s way 2
portfolio_returns = w_current.T@R
worst_return = min(portfolio_returns)
m_2 = portfolio_returns.T@p
s2_2 = sum((portfolio_returns[s]-m_2)**2*p[s] for s in range(len(p)))

#Print A.1
print("\n========== PART A.1: RESULTS ==========\n")

# Expected returns
print("Expected returns per asset:")
for i in range(len(assets)):
    print(f"{assets[i]:<6}: {mu[i]:>8.2f}%")

# Standard deviations
print("\nStandard deviations:")
for i in range(len(assets)):
    print(f"{assets[i]:<6}: {sig[i]:>8.2f}%")

# Covariance matrix
print("\nCovariance matrix:")
print(f"{'':>6}" + "".join(f"{a:>10}" for a in assets))
for i in range(len(assets)):
    print(f"{assets[i]:<6}" + "".join(f"{cov[i,j]:>10.2f}" for j in range(len(assets))))

# Portfolio results
print("\n========== PORTFOLIO RESULTS ==========")
print(f"Expected return (Method 1): {m_1:.2f}%")
print(f"Expected return (Method 2): {m_2:.2f}%")
print(f"Variance (Method 1):        {s2_1:.4f}")
print(f"Variance (Method 2):        {s2_2:.4f}")
print(f"Standard deviation:         {np.sqrt(s2_1):.2f}%")
print(f"Worst-case return:          {worst_return:.2f}%")

print("\n=======================================")

"""
A.2
"""

# Maximize expected return subject to full investment and asset-class limits.
a2_result = linprog(
    c=-mu,
    A_eq=np.ones((1, len(assets))),
    b_eq=[1.0],
    bounds=[(0.0, float(limit)) for limit in ub],
    method="highs",
)
if not a2_result.success:
    raise RuntimeError(f"A.2 optimization failed: {a2_result.message}")

w_opt = a2_result.x
m_opt = mu @ w_opt

print("\n========== PART A.2: EXPECTED VALUE MODEL ==========")
print("Optimal portfolio weights:")
for i, asset in enumerate(assets):
    print(f"{asset:<6}: {w_opt[i]:.2f}")
print(f"Maximum expected return: {m_opt:.2f}%")
print("Binding policy limits:")
for i, asset in enumerate(assets):
    if np.isclose(w_opt[i], 0.0):
        print(f"{asset}: lower bound (0%)")
    if np.isclose(w_opt[i], ub[i]):
        print(f"{asset}: upper bound ({ub[i]:.0%})")
print("The model holds no GOVT bonds, as they have the lowest expected return and are not needed to satisfy any constraints.")

"""
A.3
"""
lambdas=np.array([1,2,5,10])*0.01

constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
bounds = [(0, ub[i]) for i in range(len(assets))]

w_opt = list()
for lam in lambdas:

    def objective(w):
        m = mu @ w
        s2 = w.T @ cov @ w
        return -(m - lam * s2)

    result = minimize(
        objective,
        x0 = w_current,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Optimal weights
    w_opt.append(result.x)

print("\n========== A.3: MEAN-VARIANCE OPTIMIZATION ==========\n")

# Header
print(f"{'Lambda':>8} {'GOVT':>10} {'CORP':>10} {'EQTY':>10} {'INFR':>10} {'Return':>12} {'Variance':>12} {'Std Dev':>10}")
print("-" * 96)

# Print results for each lambda
for i in range(len(lambdas)):

    w = w_opt[i]

    m = mu @ w
    s2 = w.T @ cov @ w
    sig = np.sqrt(s2)

    print(
        f"{lambdas[i]:>8.3f} "
        f"{w[0]*100:>9.2f}% "
        f"{w[1]*100:>9.2f}% "
        f"{w[2]*100:>9.2f}% "
        f"{w[3]*100:>9.2f}% "
        f"{m:>11.2f}% "
        f"{s2:>12.4f} "
        f"{sig:>9.2f}%"
    )

print("-" * 96)

