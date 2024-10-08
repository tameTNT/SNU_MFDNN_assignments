import numpy as np
import matplotlib.pyplot as plt


def SGD(X, Y, ftn, grad, iter, alpha, batch_size, p, N):
    theta = np.random.randn(p)
    loss = [st_mean(X, Y, theta, ftn, range(N))]
    for _ in range(iter):
        indices = np.random.randint(N, size=batch_size)
        theta = theta - alpha * st_mean(X, Y, theta, grad, indices)
        loss.append(st_mean(X, Y, theta, ftn, range(N)))
    return theta, loss


# logistic loss function for a pair of x_i and y_i
def logistic_ftn(x, y, theta):
    return np.log(1 + np.exp(-y * np.dot(x, theta)))


# logistic gradient function for a pair of x_i and y_i
def grad_ftn(x, y, theta):
    expYX = np.exp(-y * np.dot(x, theta))
    return -y * expYX / (1 + expYX) * x



# stochastic mean on given function on chosen indices
def st_mean(X, Y, theta, ftn, index):
    return np.mean([ftn(X[:, i], Y[i], theta) for i in index], axis=0)


# calculate phi(X) given data X
def phi(x):
    return np.array([1, x[0], x[0]**2, x[1], x[1]**2])

def prob3():
    # data generation
    N = 30
    np.random.seed(0)
    X = np.random.randn(2, N)
    y = np.sign(X[0, :] ** 2 + X[1, :] ** 2 - 0.7)
    theta = 0.5
    c, s = np.cos(theta), np.sin(theta)
    X = np.array([[c, -s], [s, c]]) @ X
    X = X + np.array([[1], [1]])

    # apply SGD
    iter = 5000
    alpha = 0.1     # SGD stepsize
    batch_size = 15

    phiX = np.array([phi(X[:,i]) for i in range(N)]).T

    w, loss = SGD(phiX, y, logistic_ftn, grad_ftn, iter, alpha, batch_size, 5, N)

    # plot the decision boundary
    xx = np.linspace(-4, 4, 1024)
    dd = (w[3] ** 2 - 4 * w[4] * (w[0] + xx * (w[1] + w[2] * xx)))
    yy = (-w[3] + np.sqrt(dd[dd >= 0])) / (2 * w[4])
    plt.plot(xx[dd >= 0], yy, color='green')
    yy = (-w[3] - np.sqrt(dd[dd >= 0])) / (2 * w[4])
    plt.plot(xx[dd >= 0], yy, color='green')

    # plot the data points
    plt.plot(X[0, y < 0], X[1, y < 0], color='red', marker='.', linestyle="None")
    plt.plot(X[0, y > 0], X[1, y > 0], color='blue', marker='.', linestyle="None")

    plt.savefig('prob3.png')
    print(w)

prob3()
