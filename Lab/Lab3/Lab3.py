"""
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
mu = R@p # expected returns
print(mu)