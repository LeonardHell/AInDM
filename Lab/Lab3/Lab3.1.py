

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

print("The fund gives up ",(m(6)-m(4))*100000,"Euros expected profit and removes",np.sqrt((s(6)-s(4)))*100000,"Euros of variance for lambda = 0.02")
