import numpy as np
import matplotlib.pyplot as plt

# implement minibatch SGD with replacement
def SGD(X, Y, ftn, grad, iter, alpha, batch_size, p, N):
    # parameter theta
    theta = np.random.randn(p)
    # calculate the loss (over all training data)
    loss = [st_mean(X, Y, theta, ftn, range(N))]
    for _ in range(iter):
        # pick random indices for minibatch
        indices = np.random.randint(N, size=batch_size)
        # apply SGD
        theta = theta - alpha * st_mean(X, Y, theta, grad, indices)
        # store loss value
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


def prob1():
    # data generation
    N, p = 30, 20
    np.random.seed(0)
    X = np.random.randn(p, N)
    Y = 2 * np.random.randint(2, size=N) - 1

    # apply SGD
    iter = 3000
    alpha = 0.01    # SGD stepsize
    batch_size = 15

    theta, loss = SGD(X, Y, logistic_ftn, grad_ftn, iter, alpha, batch_size, p, N)

    # plot the losses
    plt.plot(range(iter + 1), loss)
    plt.savefig('prob1.png')

    return theta


print(prob1())
