""""
Lab 3
""" 
#Import everything
from pathlib import Path

import matplotlib.pyplot as plt
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
w_expected = w_opt.copy()

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
lambdas = np.array([0.01, 0.02, 0.05, 0.10])

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0, ub[i]) for i in range(len(assets))]

mean_variance_weights = []
for lam in lambdas:
    def objective(w):
        return -(mu @ w - lam * (w @ cov @ w))

    result = minimize(
        objective,
        x0=w_current,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        raise RuntimeError(f"A.3 optimization failed for lambda={lam}: {result.message}")
    mean_variance_weights.append(result.x)

print("\n========== A.3: MEAN-VARIANCE OPTIMIZATION ==========\n")

print(
    f"{'Lambda':>8} {'GOVT':>9} {'CORP':>9} {'EQTY':>9} {'INFR':>9} "
    f"{'m (%)':>10} {'s (%)':>10} {'Worst (%)':>11}"
)
print("-" * 88)

portfolio_metrics = []
for lam, weights in zip(lambdas, mean_variance_weights):
    mean = mu @ weights
    std_dev = np.sqrt(weights @ cov @ weights)
    worst = np.min(weights @ R)
    portfolio_metrics.append((mean, std_dev, worst))
    print(
        f"{lam:>8.2f} "
        + "".join(f"{weight * 100:>8.2f}% " for weight in weights)
        + f"{mean:>9.2f}% {std_dev:>9.2f}% {worst:>10.2f}%"
    )
print("-" * 88)

current_std = np.sqrt(w_current @ cov @ w_current)
current_worst = np.min(w_current @ R)
expected_std = np.sqrt(w_expected @ cov @ w_expected)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(current_std, m_1, marker="X", s=100, label="Current portfolio")
ax.scatter(expected_std, m_opt, marker="*", s=160, label="Expected-value optimum")
for lam, (mean, std_dev, _) in zip(lambdas, portfolio_metrics):
    ax.scatter(std_dev, mean, s=75, label=f"Mean-variance, $\\lambda={lam:.2f}$")
ax.set_xlabel("Standard deviation, $s$ (%)")
ax.set_ylabel("Expected return, $m$ (%)")
ax.set_title("Mean-variance portfolios")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# Resolve output relative to this file when run as a script; in interactive
# sessions, where __file__ is unavailable, use the current working directory.
script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
results_dir = script_dir / "Lab3"/"lab_results"
results_dir.mkdir(exist_ok=True)
plot_path = results_dir / "mean_variance_portfolios.png"
fig.savefig(plot_path, dpi=170)
plt.close(fig)
print(f"Plot saved to: {plot_path}")

beating_current = [
    lam
    for lam, (mean, std_dev, _) in zip(lambdas, portfolio_metrics)
    if mean > m_1 and std_dev < current_std
]
print(
    "Current portfolio is not mean-variance efficient: "
    + (
        "lambda=" + ", ".join(f"{lam:.2f}" for lam in beating_current)
        + " has both higher expected return and lower risk."
        if beating_current
        else "none of the tested portfolios has both higher expected return and lower risk."
    )
)

"""
Part B:
B.1
"""

print("\n========== B.1: SCENARIO-BY-SCENARIO OPTIMIZATION ==========\n")
scenario_results = []
for scenario_index, scenario in enumerate(scenarios):
    scenario_returns = R[:, scenario_index]
    result = linprog(
        c=-scenario_returns,
        A_eq=np.ones((1, len(assets))),
        b_eq=[1.0],
        bounds=[(0.0, float(limit)) for limit in ub],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"B.1 optimization failed for scenario {scenario}: {result.message}"
        )

    weights = result.x.copy()
    weights[np.isclose(weights, 0.0, atol=1e-9)] = 0.0
    value = scenario_returns @ weights
    fill_order = np.argsort(-scenario_returns)
    fill_rule = " -> ".join(
        f"{assets[i]} ({scenario_returns[i]:.0f}%)" for i in fill_order
    )
    scenario_results.append((scenario, weights, value, fill_rule))

print(
    f"{'Scenario':<10} {'GOVT':>9} {'CORP':>9} {'EQTY':>9} {'INFR':>9} "
    f"{'v* (%)':>10}"
)
print("-" * 60)
for scenario, weights, value, _ in scenario_results:
    print(
        f"{scenario:<10} "
        + "".join(f"{weight * 100:>8.2f}% " for weight in weights)
        + f"{value:>9.2f}%"
    )
print("-" * 60)
print("Rule: fill the portfolio in descending order of scenario returns,")
print("respecting each asset's upper bound, until the 100% budget is invested.")
for scenario, _, _, fill_rule in scenario_results:
    print(f"{scenario}: {fill_rule}")
