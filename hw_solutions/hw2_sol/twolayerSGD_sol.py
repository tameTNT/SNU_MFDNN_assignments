import numpy as np
import matplotlib.pyplot as plt

def f_true(x) :
    return (x-2)*np.cos(x*4)

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x) :
    return sigmoid(x) * (1 - sigmoid(x))

K = 10000
alpha = 0.007
N, p = 30, 50
np.random.seed(0)
a0 = np.random.normal(loc = 0.0, scale = 4.0, size = p)
b0 = np.random.normal(loc = 0.0, scale = 4.0, size = p)
u0 = np.random.normal(loc = 0, scale = 0.05, size = p)
theta = np.concatenate((a0,b0,u0))


X = np.random.normal(loc = 0.0, scale = 1.0, size = N)
Y = f_true(X)

def f_th(theta, x) :
    return np.sum(theta[2*p : 3*p] * sigmoid(theta[0 : p] * np.reshape(x,(-1,1)) + theta[p : 2*p]), axis=1)

# returns gradient of f with respect to the parameter theta = (a, b, u)
def diff_f_th(theta, x) :
    # concatenate derivatives (gradient)
    der = np.zeros(3*p)
    a = theta[0 : p]
    b = theta[p : 2*p]
    u = theta[2*p : 3*p]

    # calculate derivative (gradient) of f with respect to each parameter a, b, u
    der[:p] = u * sigmoid_prime(a * x + b) * x
    der[p:2*p] = u * sigmoid_prime(a * x + b)
    der[2*p:] = sigmoid(a * x + b)

    return der

# plot data points and true f^\star
xx = np.linspace(-2,2,1024)
plt.plot(X,f_true(X),'rx',label='Data points')
plt.plot(xx,f_true(xx),'r',label='True Fn')

# apply SGD
for k in range(K) :
    i_k = np.random.randint(N)
    theta = theta - alpha * (f_th(theta,X[i_k]) - Y[i_k]) * diff_f_th(theta, X[i_k])
    if (k+1)%2000 == 0 :
        plt.plot(xx,f_th(theta, xx),label=f'Learned Fn after {k+1} iterations')

plt.legend()
# plt.show()
plt.savefig('twolayerSGD_plot.png')
