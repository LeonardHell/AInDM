import math
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
GROUP_MODES = ("snake", "random", "best_vs_best")
KNOCKOUT_MODES = ("best_vs_best", "random", "best_vs_worst")
ns = [4, 8, 16, 32]
def LoadData():
    """ Load data from simulation """
    
    with open("Project_2/tournament_results_lesssimul.txt", "r") as file:
        data = json.load(file)
    return data
data = LoadData() #results,(n), x , Gr-mode, QperG, KO-mode, probabilities

def Probabilities(x,Grmode,KOmode, QperG = 2,n=16):
    x_prob = data[str(n)]["results"][str(x)][Grmode][str(QperG)][KOmode]
    return x_prob

def ProbStrongest(probs):
    return probs[0]

def NumberofGames(x,n=16,QperG = 2):
    sum=0
    for i in range(x):
        sum+=2**i
    return sum+math.factorial(int(n/2**x*QperG))

def Predictability(probs):
    distribution = probs
    return np.var(distribution)

def det_hyper_params(results) -> dict:
    evaluation = copy.deepcopy(results)
    for n in ns:
        for x in range(int(np.log2(n))):
            for Grmode in GROUP_MODES:
                for i in range(x+1):
                    QperG = 2**(i)
                    for KOmode in KNOCKOUT_MODES:
                        probs = evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode]
                        evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode] = {"probs":probs, "num_games":NumberofGames(x, n, QperG), "predictability":Predictability(probs), "strongest_prob":ProbStrongest(probs)}
    return evaluation

evaluation = det_hyper_params(data)
def Pareto_dom (n,para_results):
    """return True if param_results is Pareto dominant regarding the three goals and False if not"""
    for x in range(int(np.log2(n))):
            for Grmode in GROUP_MODES:
                for i in range(x+1):
                    QperG = 2**(i)
                    for KOmode in KNOCKOUT_MODES:
                        test_para = evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode]
                        if test_para["num_games"]< para_results["num_games"] and np.round(test_para["strongest_prob"],4)>np.round(para_results["strongest_prob"],4) and np.round(test_para["predictability"],4)<np.round(para_results["predictability"],4):
                            # So if test_para dominates param_results for all three goals
                            return False
    #No dominant parameter set found
    return True

def Pareto_Front (n):
    Pareto_front = copy.deepcopy(evaluation["results"][str(n)])
    Pareto_front_parameters=[]
    for x in range(int(np.log2(n))):
        for Grmode in GROUP_MODES:
            for i in range(x+1):
                QperG = 2**(i)
                for KOmode in KNOCKOUT_MODES:
                    param = Pareto_front[str(x)][Grmode][str(QperG)][KOmode]
                    if not Pareto_dom(n,param):
                        del Pareto_front[str(x)][Grmode][str(QperG)][KOmode]
                    else:
                        Pareto_front_parameters.append([[str(x)],[Grmode],[str(QperG)],[KOmode]])
                    if Pareto_front[str(x)][Grmode][str(QperG)]=={}:
                        del Pareto_front[str(x)][Grmode][str(QperG)]
                    if Pareto_front[str(x)][Grmode]=={}:
                        del Pareto_front[str(x)][Grmode]
                    if Pareto_front[str(x)]=={}:
                        del Pareto_front[str(x)]
    return Pareto_front,Pareto_front_parameters
print(Pareto_Front(4)[0]["0"])

def Plotdata(function):
    x = np.array([1,2,3,4])
    y1a = [function(i, "snake", "best_vs_best") for i in x]
    y1b = [function(i, "random", "best_vs_best") for i in x]
    y1c = [function(i, "best_vs_best", "best_vs_best") for i in x]
    y2a = [function(i, "snake", "random") for i in x]
    y2b = [function(i, "random", "random") for i in x]
    y2c = [function(i, "best_vs_best", "random") for i in x]
    y3a = [function(i, "snake", "best_vs_worst") for i in x]
    y3b = [function(i, "random", "best_vs_worst") for i in x]
    y3c = [function(i, "best_vs_best", "best_vs_worst") for i in x]

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
    plt.ylabel(str(function))
    plt.title(str(function))
    plt.legend()
    # Display plot
    plt.show()
Plotdata(ProbStrongest)
Plotdata(Predictability)