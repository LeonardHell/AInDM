""""
Lab 3
""" 
#Import everything
import numpy as np

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
        print(i,j)
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
print("\n========== PART A: RESULTS ==========\n")

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