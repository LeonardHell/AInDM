import numpy as np
import matplotlib.pyplot as plt

"""
Part A: route1, moments
"""
print("Part A: route1, moments")
#A.1
print("A.1")
def Pi(x,sc):
    if sc == 'S':
        return 70 - 8*x
    elif sc == 'N':
        return 20 + 2*x
    elif sc == 'B':
        return 10 + 12*x
print("Pi(x,S)= 70-8x")
print("Pi(x,N)= 20+2x")
print("Pi(x,B)= 10+12x")

#A.2
print("A.2")
def m(x):
    return 30 + 2 * x
x = np.linspace(0, 6, 100)

def s2(x):
    return 550 - 300*x + 50*x**2

def Deviat(x,sc):
    return Pi(x,sc) - m(x)

print("m(x)= 30+2x")
print("s^2(x)= 550-300x+50x^2")
print("Pi(x,S)-m(x)= 40-10x")
print("Pi(x,N)-m(x)= -10")
print("Pi(x,B)-m(x)= -20+10x")  

#A.3
print("A.3")
print(f"Max return is for x=6, with a return of {m(6)} and a variance of {s2(6)}")

#A.4
print("A.4")
print("for lambda = 0.02, the optimal x is 4, with a return of", m(4), "and a variance of", s2(4))
print("for lambda = 0.01, the optimal x is 5, with a return of", m(5), "and a variance of", s2(5))
print("So lambda= 0.01, i.e. x= 5, has a higher variance")
x = np.linspace(0, 6, 100)
y = m(x)
z = s2(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter([4, 5], [38, 40], [150, 300], color='red', s=50, label='Decision Points')

ax.plot(x, y, z)

ax.set_xlabel('x')
ax.set_ylabel('m')
ax.set_zlabel('s^2')
plt.legend()
plt.show()
print("The fund gives up ",(m(6)-m(4))*100/m(6),"percent expected profit and removes",((s2(6)-s2(4)))*100/s2(6),"percent of variance for lambda = 0.02")


#B:route 2
print("Part B: route 2,scenarios")
#B1
print("B.1")
x_max_S = 0
x_max_N = 6
x_max_B = 6
print("x^* for S,N,B is ", x_max_S, x_max_N, x_max_B)
Pi_max_S = 70
Pi_max_N = 32
Pi_max_B = 82
print("Expected maximum return for S, N, B is ", Pi_max_S, Pi_max_N, Pi_max_B)

#B2
print("B.2")
Prob_xmax0 = 1/4
Prob_xmax6 = 1/4+1/2
print("The probability of x_max = 0 is ", Prob_xmax0)
print("The probability of x_max = 6 is ", Prob_xmax6)
Expected_xmax= Prob_xmax0*0 + Prob_xmax6*6
print("The expected value of x_max is ", Expected_xmax)
Var_xmax = Prob_xmax0*(0 - Expected_xmax)**2 + Prob_xmax6*(6 - Expected_xmax)**2
print("The variance of x_max is ", Var_xmax)
Expected_Pi_max = 1/4*Pi_max_S + 1/2*Pi_max_N+1/4*Pi_max_B
print("The expected maximum return is ", Expected_Pi_max)

#B3
print("B.3")
# value of perfect information
VPI = Expected_Pi_max - m(6)
print("The value of perfect information is ", VPI*100000,"Euros")

#Evaluation of expected profit of E[x^*]
print("The expected profit of E[x^*] is", m(Expected_xmax)*100000,"Euros")

#B4
print("B.4")
print("""
THE TWO ROUTES COMPARED

route 1  expected value  x = 6, return 42 percent              

route 1  mean-variance   x = 4,5 (lambda=0.02,0.01)                      

route 2  x*(ω)           0 or 6                      

route 2  E[v*]           54 percent                           
         (perfect information)

         value of perfect
         information     12 percent                  

route 2  plug in E[x*]   x = 4.5: return 39 percent (<42)    
""")

"""
Part C: two objectives, profit and CO_2
"""
print("Part C: two objectives, profit and CO_2")

#C.1
print("C.1")
def e(x, sc):
    if sc == 'S':
        return 10 + x
    elif sc == 'N':
        return 10 + 3*x
    elif sc == 'B':
        return 10 + 5*x

print("e(x,S) = 10+x")
print("e(x,N) = 10+3x")
print("e(x,B) = 10+5x")

def m_CO2(x):
    return 10 + 3*x
print("m_CO2(x) = 10+3x")
print("So every decision (0=<x<=6) is E-efficient for max return and min CO2 emissions, since the return increases with x, but the CO2 emissions also increase with x")

#C.2
print("C.2")
print("E[pi]-tau*E[e] = 30-10*tau + (2-3*tau)*x")
print("for tau = 0.5, the optimal x is 6")
print("for tau= 1, the optimal x is 0")
print("the switch is at tau = 2/3, so a price of ",round((2/3)*100000,2),"Euros per ton CO2 emissions")

#C.3
print("C.3")

print("S: Eps(x) = {0}")
print("N: Eps(x) = {0,1,2,3,4,5,6}")
print("B: Eps(x) = {0,1,2,3,4,5,6}")
print("So probability of Eps(x) = {0} is 1/4, and probability of Eps(x) = {0,1,2,3,4,5,6} is 3/4")
print("So x= 0 is efficient whatever happens")

#C.4
print("C.4")
print("S: pi(S)-tau*e(S) = 70-10*tau + (-8-1*tau)*x")
print("N: pi(N)-tau*e(N) = 20-10*tau + (2-3*tau)*x")
print("B: pi(B)-tau*e(B) = 10-10*tau + (12-5*tau)*x")
print("""
tau     0.5     1     3
----------------------------
S       x*=0     x*=0   x*=0
N       x*=6     x*=0   x*=0
B       x*=6     x*=6   x*=0
P(x*=6)  3/4      1/4     0
E[x*]    4.5      1.5     0
""")

"""
Part D: robust models, no probabilities
"""
#D.1
print("D.1")
print("""The min(x,w) = scenario S   for  x = 5,6
                        scenario N   for  x = 1,2,3,4,5   
                        scenario B   for  x = 0,1 """)

x = np.linspace(0, 6, 100)
def mini(x):
    return np.minimum(np.minimum(Pi(x,"S"), Pi(x,"N")), Pi(x,"B"))
minimum = mini(x)

plt.plot(x, Pi(x,"S"), label="Slowdown")
plt.plot(x, Pi(x,"N"), label="Normal")
plt.plot(x, Pi(x,"B"), label="Boom")
plt.plot(x, minimum, "k--", linewidth=3, label="Minimum (lower envelope)")

plt.xlabel("x")
plt.ylabel("Profit")
plt.title("Three Profit Functions and Their Lower Envelope")
plt.legend()
plt.grid()
plt.show()
print("The lower envelope increases until x= 5, than decreases, so the optimal is x = 5, with a return of", round(Pi(5,"S")*100000,2),"Euros")
print("""As an LP: max w s.t. w ≤ 70 − 8x,w ≤ 20 + 2x, w ≤ 10 + 12x""")

#D.2
print("D.2")
print(f"""
decision          x       worst       E[pi]
-----------------------------------------------------
Expect-value      6      {mini(6)}    {m(6)}
mean-var (0.02)   4      {mini(4)}    {m(4)}
robust            5      {mini(5)}    {m(5)}
""")
print(f"The robustness cost is {m(6)-m(5)}*100k of expected profit")
print(f"It raises the lowest profit from {mini(6)}100k to {mini(5)}100k")

#D.3
print("D.3")