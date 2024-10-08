import numpy as np
import matplotlib.pyplot as plt


np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2) 

x = np.linspace(-5,20,100)
plt.plot(x,f(x), 'k')

alpha_list = [0.01, 0.3, 4]
limit_pts = [[],[],[],[]]

for i,alpha in enumerate(alpha_list):
    for _ in range(20):
        x = np.random.uniform(-5,20)
        for _ in range(1000):
            x -= alpha*fprime(x)
        limit_pts[i].append(x)

x = np.array(limit_pts[0])
plt.plot(x,f(x), 'g.', label=f'Limit points with \alpha={alpha_list[0]}$', markersize=15)
x = np.array(limit_pts[1])
plt.plot(x,f(x), 'rx', label=f'Limit points with \alpha={alpha_list[1]}$')
print(f'The limit points with alpha={alpha_list[2]}:')
print(limit_pts[2])
print('These are all NaNs, which idicate divergence.')
plt.show()
