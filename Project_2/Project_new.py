import math
import numpy as np
import json
GROUP_MODES = ("snake", "random", "best_vs_best")
KNOCKOUT_MODES = ("best_vs_best", "random", "best_vs_worst")

def LoadData():
    """ Load data from simulation """
    
    with open("Project_2/tournament_results.txt", "r") as file:
        data = json.load(file)
    return data
data = LoadData() #results,(n), x , Gr-mode, QperG, KO-mode, probabilities

def Probabilities(x,Grmode,KOmode, QperG = 2):
    x_prob = data["results"][str(x)][Grmode][str(QperG)][KOmode]
    return x_prob

def ProbStrongest(x,Grmode,KOmode, QperG = 2):
    return Probabilities(x,Grmode,KOmode, QperG = 2)[0]

def NumberofGames(x,n=16,QperG = 2):
    sum=0
    for i in range(x):
        sum+=2**i
    return sum+math.factorial(n/2**(x-2))

def Predictability(x,Grmode,KOmode, QperG = 2):
    distribution = Probabilities(x,Grmode,KOmode, QperG = 2)
    return np.var(distribution)

import matplotlib.pyplot as plt

# Data
x = np.array([1,2,3,4])
y1a = [ProbStrongest(i, "snake", "best_vs_best") for i in x]
y1b = [ProbStrongest(i, "random", "best_vs_best") for i in x]
y1c = [ProbStrongest(i, "best_vs_best", "best_vs_best") for i in x]
y2a = [ProbStrongest(i, "snake", "random") for i in x]
y2b = [ProbStrongest(i, "random", "random") for i in x]
y2c = [ProbStrongest(i, "best_vs_best", "random") for i in x]
y3a = [ProbStrongest(i, "snake", "best_vs_worst") for i in x]
y3b = [ProbStrongest(i, "random", "best_vs_worst") for i in x]
y3c = [ProbStrongest(i, "best_vs_best", "best_vs_worst") for i in x]

#print(y2)

# Create plot
plt.plot(x, y1a, label="best vs best",linestyle="--",color = "blue")
plt.plot(x, y1b, label="best vs best",linestyle="-",color = "blue")
plt.plot(x, y1c, label="best vs best",linestyle="-.",color = "blue")
plt.plot(x, y2a,label = "random",linestyle="--",color = "green")
plt.plot(x, y2b,label = "random",linestyle="-",color = "green")
plt.plot(x, y2c,label = "random",linestyle="-.",color = "green")
plt.plot(x, y3a, label = "best_vs_worst",linestyle="--",color = "red")
plt.plot(x, y3b, label = "best_vs_worst",linestyle="-",color = "red")
plt.plot(x, y3c, label = "best_vs_worst",linestyle="-.",color = "red")

# Add labels and title
plt.xlabel("X values")
plt.ylabel("Strongest wins prob")
plt.title("Group phase snake")
plt.legend()
# Display plot
plt.show()