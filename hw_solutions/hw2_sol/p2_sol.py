import numpy as np
import matplotlib.pyplot as plt

# check whether SGD has encountered a point of non-differentiability or not
checker = True

def SGD(X, Y, ftn, grad, iter, alpha, batch_size, p, N, lam):
    theta = np.random.randn(p)
    loss = [st_mean(X, Y, theta, ftn, range(N)) + lam * np.dot(theta, theta)]
    for _ in range(iter):
        indices = np.random.randint(N, size=batch_size)
        theta = theta - alpha * (st_mean(X, Y, theta, grad, indices) + 2 * lam * theta)
        loss.append(st_mean(X, Y, theta, ftn, range(N)) + lam * np.dot(theta, theta))
    return theta, loss


# ReLU loss function for a pair of x_i and y_i
def ReLU_ftn(x, y, theta):
    return np.maximum(0, 1 - y * np.dot(x, theta))


# ReLU subgradient function for a pair of x_i and y_i
def grad_ftn(x, y, theta):
    if 1 - y * np.dot(x, theta) == 0 :
        global checker
        checker = False
        print("SGD encountered a point of non-differentiability")
    grad = -y * x
    return (grad) if (1 - y * np.dot(x, theta) > 0) else (0.0 * grad)


# stochastic mean on given function on chosen indices
def st_mean(X, Y, theta, ftn, index):
    return np.mean([ftn(X[:, i], Y[i], theta) for i in index], axis=0)


def prob2():
    # data generation
    N, p = 30, 20
    np.random.seed(0)
    X = np.random.randn(p, N)
    Y = 2 * np.random.randint(2, size=N) - 1

    # apply SGD
    iter = 3000
    alpha = 0.01    # SGD stepsize
    batch_size = 15
    lam = 0.1

    theta, loss = SGD(X, Y, ReLU_ftn, grad_ftn, iter, alpha, batch_size, p, N, lam)

    # plot the losses
    plt.plot(range(iter + 1), loss)
    plt.savefig('prob2.png')

    if checker == True :
        print("SGD never encountered a point of non-differentiability")

    return theta


print(prob2())
