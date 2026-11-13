

import numpy as np
import matplotlib.pyplot as plt
def m(x):
    return 30 + 2 * x
x = np.linspace(0, 6, 100)

def s(x):
    return 550 - 300*x + 50*x**2

x = np.linspace(0, 6, 100)

y = m(x)
z = s(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter([4, 5], [38, 40], [150, 300], color='red', s=50, label='Decision Points')

ax.plot(x, y, z)

ax.set_xlabel('x')
ax.set_ylabel('m')
ax.set_zlabel('s^2')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def m(x):
    return 30 + 2 * x
x = np.linspace(0, 6, 100)

def s(x):
    return 550 - 300*x + 50*x**2

x = np.linspace(0, 6, 100)

y = m(x)
z = s(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter([4, 5], [38, 40], [150, 300], color='red', s=50, label='Decision Points')

ax.plot(x, y, z)

ax.set_xlabel('x')
ax.set_ylabel('m')
ax.set_zlabel('s^2')
plt.legend()
plt.show()

print("The fund gives up ",(m(6)-m(4))*100/m(6),"percent expected profit and removes",((s(6)-s(4)))*100/s(6),"percent of variance for lambda = 0.02")


#B:route 2
#B1
x_max_S = 0
x_max_N = 6
x_max_B = 6
print("x^* for S,N,B is ", x_max_S, x_max_N, x_max_B)
Pi_max_S = 70
Pi_max_N = 32
Pi_max_B = 82
print("Expected maximum return for S, N, B is ", Pi_max_S, Pi_max_N, Pi_max_B)

#B2
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
# value of perfect information
VPI = Expected_Pi_max - m(6)
print("The value of perfect information is ", VPI*100000,"Euros")

#Evaluation of expected profit of E[x^*]
print("The expected profit of E[x^*] is", m(Expected_xmax)*100000,"Euros")

#B4