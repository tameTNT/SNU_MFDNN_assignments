import torch
import math

lr = 1e-2
B = 16
iterations = 200

"""Log-derivative trick"""

mu = torch.tensor([[0.]])
tau = torch.tensor([[0.]])

history1 = torch.zeros((iterations+1, 2))

for itr in range(iterations):
    X = torch.normal(0,1,size=(B,1))*tau.exp() + mu
    g_mu = torch.sum(X * X.sin() * (X-mu) / (2*tau).exp() + mu - 1 ,dim=0) / B
    g_tau = torch.sum(X * X.sin() * (-1 + (X-mu)**2 / (2*tau).exp()) + tau.exp() -1,dim=0) / B
    mu -= lr*g_mu
    tau -= lr*g_tau
    # save history
    history1[itr+1][0] = mu
    history1[itr+1][1] = tau
    
print(mu, tau)

"""Reparameterization trick"""

mu = torch.tensor([[0.]])
tau = torch.tensor([[0.]])

history2 = torch.zeros((iterations+1, 2))

for itr in range(iterations):
    Z = torch.normal(0,1,size=(B,1))
    Y = Z * tau.exp() + mu
    g_mu = torch.sum(Y * Y.cos() + Y.sin() + mu - 1 ,dim=0) / B
    g_tau = torch.sum(tau.exp() * Z * (Y * Y.cos() + Y.sin())  + tau.exp() -1,dim=0) / B
    mu -= lr*g_mu
    tau -= lr*g_tau
    # save history
    history2[itr+1][0] = mu
    history2[itr+1][1] = tau
    
print(mu, tau)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

x1 = np.array(history1[:, 0])
y1 = np.array(history1[:, 1])

x2 = np.array(history2[:, 0])
y2 = np.array(history2[:, 1])

plt.subplot(1, 2, 1)
plt.scatter(0,0, s=100, c='green')
#plt.scatter(c[0][0],c[0][1], s=100, c='red')
plt.plot(x1, y1, linestyle='solid',color='blue')
plt.title('log derivative trick')

plt.subplot(1, 2, 2)
plt.scatter(0,0, s=100, c='green')
#plt.scatter(c[0][0],c[0][1], s=100, c='red')
plt.plot(x2, y2, linestyle='solid',color='blue')
plt.title('reparametrization trick')

plt.tight_layout()
plt.show()