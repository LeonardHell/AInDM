import math
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
GROUP_MODES = ("snake", "random", "best_vs_best")
KNOCKOUT_MODES = ("best_vs_best", "random", "best_vs_worst")
ns = [4, 8, 16, 32, 64]
list_hyper_parameters = ["num_games", "predictability", "strongest_prob"]


def LoadData():
    """ Load data from simulation """
    
    with open("Project_2/tournament_results_n_(4, 8, 16, 32, 64)_100000.txt", "r") as file:
        data = json.load(file)
    return data
data = LoadData() #results,(n), x , Gr-mode, QperG, KO-mode, probabilities

def Probabilities(x,Grmode,KOmode, QperG = 2,n=16):
    x_prob = data[str(n)]["results"][str(x)][Grmode][str(QperG)][KOmode]
    return x_prob

def ProbStrongest(probs):
    sum_1 = 0
    for i in range(len(probs)):
        sum_1 += 1/((i+1))*probs[i]
    return sum_1

def NumberofGames(x,n=16,QperG = 2):
    sum1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(x):
        sum1 += 2**i
    for j in range(int(n/2**x*QperG)):
        sum_2 += j
    for j in range(n):
        sum_3 += j
    NumbonlyGroup = sum_3
    return (sum1 + (2**x)/QperG*sum_2),NumbonlyGroup

def Predictability(n, probs):
    distribution = probs
    return np.var(distribution)*n

def det_hyper_params(results) -> dict:
    evaluation = copy.deepcopy(results)
    for n in ns:
        for x in range(int(np.log2(n)) + 1):
            for Grmode in GROUP_MODES:
                for i in range(x+1):
                    QperG = 2**(i)
                    for KOmode in KNOCKOUT_MODES:
                        probs, tension = evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode]
                        num_g,NumbonlyGroups = NumberofGames(x, n, QperG)
                        evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode] = {"probs":probs, "num_games":num_g/NumbonlyGroups, "predictability":tension/num_g, "strongest_prob":-ProbStrongest(probs)}
    return evaluation

evaluation = det_hyper_params(data)
def Pareto_dom (n,para_results,sharp=4):
    """return True if param_results is Pareto dominant regarding the three goals and False if not"""
    for x in range(int(np.log2(n)) + 1):
            for Grmode in GROUP_MODES:
                for i in range(x+1):
                    QperG = 2**(i)
                    for KOmode in KNOCKOUT_MODES:
                        test_para = evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode]
                        #if test_para["num_games"]<= para_results["num_games"] and np.round(test_para["strongest_prob"],sharp)>=np.round(para_results["strongest_prob"],sharp) and np.round(test_para["predictability"],sharp)<=np.round(para_results["predictability"],sharp):
                        if (
                            test_para["num_games"] <= para_results["num_games"]
                            and test_para["strongest_prob"] <= para_results["strongest_prob"]
                            and test_para["predictability"] <= para_results["predictability"]
                            and (
                                test_para["num_games"] < para_results["num_games"]
                                or test_para["strongest_prob"] < para_results["strongest_prob"]
                                or test_para["predictability"] < para_results["predictability"]
                            )
                        ):
                            # So if test_para dominates param_results for all three goals
                            return False
    #No dominant parameter set found
    return True

def Pareto_Front (n):
    Pareto_front = copy.deepcopy(evaluation["results"][str(n)])
    Pareto_front_parameters=[]
    for x in range(int(np.log2(n)) + 1):
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

def Len_of_data(d):
    total_probs = 0
    for x_dict in d.values():
        for gr_dict in x_dict.values():
            for q_dict in gr_dict.values():
                for ko_dict in q_dict.values():
                    total_probs += 1
    return total_probs

#print(Pareto_Front(4)[0].values())
#print(Len_of_data(Pareto_Front(32)[0]))
#print(Len_of_data(data["results"]["32"]))

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
#Plotdata(ProbStrongest)
#Plotdata(Predictability)

def Plot_Pareto(n):
    Pareto_front = Pareto_Front(n)[0]
    num_games = []
    strongest_prob = []
    predictability = []
    for x_dict in Pareto_front.values():
        for gr_dict in x_dict.values():
            for q_dict in gr_dict.values():
                for ko_dict in q_dict.values():
                    num_games.append(ko_dict["num_games"])
                    strongest_prob.append(ko_dict["strongest_prob"])
                    predictability.append(ko_dict["predictability"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(num_games, strongest_prob, predictability)
    ax.set_xlabel("Number of games (minimize)")
    ax.set_ylabel("Strongest player's win probability (maximize)")
    ax.set_zlabel("Predictability (minimize)")
    ax.set_title(f"Pareto Front for n={n}")
    fig.tight_layout()
    plt.show()
    return fig, ax


def Weighted_Sum(lam1,lam2,n):
    """ Returns the minimum of the weighted sum as parameters and value
     Args
      lam1 : weights number of games
      lam2 : weights predictability
      """
    if not (0 < lam1 < 1) or not (0 < lam2 < 1) or not (lam1 + lam2 < 1):
        raise ValueError("lam1 and lam2 must each be >0 and <1, and their sum must be <1")
    def Objective(dict):
        lam3 = 1-lam1-lam2
        return lam1*dict["num_games"]+lam2*dict["predictability"]+lam3*dict["strongest_prob"]
    
    d = evaluation["results"][str(n)]
    minOb = None
    Front_parameters = []
    for x in range(int(np.log2(n))+1):
        for Grmode in GROUP_MODES:
            for i in range(x+1):
                QperG = 2**(i)
                for KOmode in KNOCKOUT_MODES:
                    #print(Objective(prob,x,n,QperG))
                    obje = Objective(d[str(x)][Grmode][str(QperG)][KOmode])
                    if minOb == None:
                        minOb = obje
                        Front_parameters =[[str(x)],[Grmode],[str(QperG)],[KOmode]]
                        values = d[str(x)][Grmode][str(QperG)][KOmode]
                    elif obje < minOb:
                        minOb = obje
                        Front_parameters =[[str(x)],[Grmode],[str(QperG)],[KOmode]]
                        values = d[str(x)][Grmode][str(QperG)][KOmode]
                    elif obje == minOb:
                        Front_parameters.append([[str(x)],[Grmode],[str(QperG)],[KOmode]])
                        values.append(d[str(x)][Grmode][str(QperG)][KOmode])
                                   
    return (Front_parameters,minOb,values)

def Weighted_Sum_Output(lam1,lam2,n):
    Objective = 0

    min_params, min_value = min_param("num_games", n)
    lam1 = lam1*min_value
    Objective += min_value

    min_params, min_value = min_param("num_games", n)
    lam2 = lam2*min_value
    Objective += min_value
    
    lam1,lam2 = lam1/Objective,lam2/Objective
    print("du rüssel")
    return None

def min_param(param: str, n: int):
    """return the parameter set that minimizes the given hyperparameter"""
    min_value = None
    min_parameters = []
    for x in range(int(np.log2(n)) + 1):
        for Grmode in GROUP_MODES:
            for i in range(x+1):
                QperG = 2**(i)
                for KOmode in KNOCKOUT_MODES:
                    params = evaluation["results"][str(n)][str(x)][Grmode][str(QperG)][KOmode]
                    if min_value is None or params[param] < min_value:
                        min_value = params[param]
                        min_parameters = [[str(x)], [Grmode], [str(QperG)], [KOmode]]
                    elif params[param] == min_value:
                        min_parameters.append([[str(x)], [Grmode], [str(QperG)], [KOmode]])
    return (min_parameters, min_value)

def epsilon_constraint(epsilon:list, n:int):
    """
    Epsilon constraint
    Give Vektor of limits leaving one entry equal to None
    """
    idx = epsilon.index(None)
    missing_parameter = list_hyper_parameters[idx]
    epsilon_possible = copy.deepcopy(evaluation["results"][str(n)])
    candidates = []
    min_j = None
    parameters = None
    bindingness = epsilon
    for eps in epsilon:
        if not eps == None:
            for x in range(int(np.log2(n)) + 1):
                        for Grmode in GROUP_MODES:
                            for i in range(x+1):
                                QperG = 2**(i)
                                for KOmode in KNOCKOUT_MODES:
                                    params = epsilon_possible[str(x)][Grmode][str(QperG)][KOmode]
                                    if params[list_hyper_parameters[epsilon.index(eps)]] <= eps:
                                        del epsilon_possible[str(x)][Grmode][str(QperG)][KOmode]
                                    if len(epsilon_possible) < 1:
                                        print("Impossible epsilon values")
                                        return None

    for x in epsilon_possible:
        for Grmode in epsilon_possible[x]:
            for QperG in epsilon_possible[x][Grmode]:
                for KOmode in epsilon_possible[x][Grmode][QperG]:
                    params = epsilon_possible[x][Grmode][QperG][KOmode]
                    if min_j == None:
                        min_j = params[missing_parameter]
                        parameters = [[str(x)], [Grmode], [str(QperG)], [KOmode]]
                    if min_j > params[missing_parameter]:
                        min_j = params[missing_parameter]
                        parameters = [[str(x)], [Grmode], [str(QperG)], [KOmode]]
                        for eps in epsilon:
                            if not eps == None:
                                if eps == params[list_hyper_parameters[epsilon.index(eps)]]:
                                    bindingness[epsilon.index(eps)] = True
                                else:
                                    bindingness[epsilon.index(eps)] = False
    return min_j, parameters, bindingness

# Print best obtainable hyperparameters and their corresponding parameters

for n in ns:
     for param in ["num_games", "strongest_prob", "predictability"]:
        min_params, min_value = min_param(param, n)
        print(f"Best obtainable {param} for n={n}: {min_value} with parameters: {min_params}")

# Plot Efficient Pareto Frontier
"""
for n in ns:
    Plot_Pareto(n)
"""
# Print Weighted sums
n=32
lam1 = 1/3
lam2 = 1/3
para,minob,values = Weighted_Sum(lam1,lam2,n)
print(n,para,minob)
print(f"""
numgam= {values["num_games"]}
stronprob = {values["strongest_prob"]}
tension = {values["predictability"]}""")

# Print epsilon constraints

#test
print(epsilon_constraint([0.1, 0.1, None], 16))